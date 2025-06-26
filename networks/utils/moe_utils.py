import torch
import torch.nn as nn
import math
import copy
import typing
import torch.nn.functional as F





def router_z_loss_func(router_logits: torch.Tensor) -> float:
    r"""
    Compute the router z-loss implemented in PyTorch.
    The router z-loss was introduced in [Designing Effective Sparse Expert Models](https://arxiv.org/abs/2202.08906).
    It encourages router logits to remain small in an effort to improve stability.
    Args:
        router_logits (`float`):
            Input logits of shape [batch_size, sequence_length, num_experts]
    Returns:
        Scalar router z-loss.
    """
    num_groups, H, W, _ = router_logits.shape
    log_z = torch.logsumexp(router_logits, dim=-1)
    z_loss = log_z**2
    return torch.sum(z_loss) / (num_groups * H * W)


def load_balancing_loss_func(router_probs: torch.Tensor, expert_indices: torch.Tensor) -> float:
    r"""
    Computes auxiliary load balancing loss as in Switch Transformer - implemented in Pytorch.
    See Switch Transformer (https://arxiv.org/abs/2101.03961) for more details. This function implements the loss
    function presented in equations (4) - (6) of the paper. It aims at penalizing cases where the routing between
    experts is too unbalanced.
    Args:
        router_probs (`torch.Tensor`):
            Probability assigned to each expert per token. Shape: [batch_size, seqeunce_length, num_experts].
        expert_indices (`torch.Tensor`):
            Indices tensor of shape [batch_size, seqeunce_length] identifying the selected expert for a given token.
    Returns:
        The auxiliary loss.
    """
    router_probs = router_probs.reshape(router_probs.shape[0], -1, router_probs.shape[-1])
    expert_indices = expert_indices.reshape(expert_indices.shape[0], -1)
    num_experts = router_probs.shape[-1]

    # cast the expert indices to int64, otherwise one-hot encoding will fail
    if expert_indices.dtype != torch.int64:
        expert_indices = expert_indices.to(torch.int64)

    if len(expert_indices.shape) == 2:
        expert_indices = expert_indices.unsqueeze(2)

    expert_mask = torch.nn.functional.one_hot(expert_indices, num_experts)

    # For a given token, determine if it was routed to a given expert.
    expert_mask = torch.max(expert_mask, axis=-2).values

    # cast to float32 otherwise mean will fail
    expert_mask = expert_mask.to(torch.float32)
    tokens_per_group_and_expert = torch.mean(expert_mask, axis=-2)

    router_prob_per_group_and_expert = torch.mean(router_probs, axis=-2)
    return torch.mean(tokens_per_group_and_expert * router_prob_per_group_and_expert) * (num_experts**2)


class Top1Router(nn.Module):
    """
    Router using tokens choose top-1 experts assignment.
    This router uses the same mechanism as in Switch Transformer (https://arxiv.org/abs/2101.03961) and V-MoE
    (https://arxiv.org/abs/2106.05974): tokens choose their top experts. Items are sorted by router_probs and then
    routed to their choice of expert until the expert's expert_capacity is reached. **There is no guarantee that each
    token is processed by an expert**, or that each expert receives at least one token.
    """

    def __init__(self, attr_len, hidden_size, num_experts, router_bias=True, router_noise=1e-2):
        super().__init__()
        self.num_experts = num_experts
        # self.classifier = torch.nn.Sequential(torch.nn.Linear(attr_len, hidden_size), 
        #                                         torch.nn.GELU(),
        #                                         torch.nn.LayerNorm(hidden_size), 
        #                                         torch.nn.Linear(hidden_size, self.num_experts, bias=router_bias))
        self.classifier = nn.Linear(attr_len, self.num_experts, bias=router_bias)
        self.jitter_noise = router_noise

    def _compute_router_probabilities(self, attr: torch.Tensor):
        r"""
        Computes router probabilities from input hidden states.
        Args:
            attr (`torch.Tensor`):
                (batch_size, sequence_length, hidden_dim) from which router probabilities are computed.
        Returns:
            router_probabilities (`torch.Tensor`):
                Tensor of shape (batch_size, sequence_length, num_experts) corresponding to the probabilities for each
                token and expert. Used for routing tokens to experts.
            router_logits (`torch.Tensor`):
                Logits tensor of shape (batch_size, sequence_length, num_experts) corresponding to raw router logits.
                This is used later for computing router z-loss.
        """
        # float32 is used to ensure stability. See the discussion of "selective precision" in
        # https://arxiv.org/abs/2101.03961.
        # We also store the previous dtype to cast back the output to the previous dtype

        if self.jitter_noise > 0:
            # Get the lower and upper bound of the uniform distribution
            # Adapted from: https://stackoverflow.com/questions/44328530/how-to-get-a-uniform-distribution-in-a-range-r1-r2-in-pytorch
            distrib_lower_bound = 1.0 - self.jitter_noise
            distrib_upper_bound = 1.0 + self.jitter_noise

            uniform_distrib = torch.rand(attr.shape, device=attr.device)
            uniform_distrib = uniform_distrib * (distrib_lower_bound - distrib_upper_bound)

            uniform_distrib = uniform_distrib + distrib_upper_bound
            # Multiply the token inputs by the uniform distribution - adding some noise
            new_attr = attr * uniform_distrib

        # Shape: [num_groups, tokens_per_group, num_experts]
        # self._cast_classifier()
        router_logits = self.classifier(new_attr)

        # Apply Softmax and cast back to the original `dtype`
        router_probabilities = nn.functional.softmax(router_logits, dim=-1)
        return router_probabilities, router_logits


    def forward(self, hidden_states: torch.Tensor):
        r"""
        Generic forward function for every Router class. Each Router expects to have the same input hidden states
        (`hidden_states`) corresponding to the hidden states for each token, the `expert_capacity` corresponding to the
        number of tokens the Router will send to each expert, some Routers can send up to few tokens to each expert.
        Each Router works as the following: it expects the hidden states for each token, gets the `router_probs` and
        `router_logits` from the `router_weights`. This will assign for each token, the raw probability to be assigned
        to an expert. Then each Router class will have to define its own `_compute_routing_instructions`.
        Args:
            hidden_states (`torch.Tensor`) :
                [num_groups, tokens_per_group, hidden_dim] inputs to send to experts.
        Returns:
            Tuple[`torch.Tensor`, `torch.Tensor`, `torch.Tensor`] Tuple containing the expert index, the router probs
            and the router logits. The router probabilities and logits are required to compute the loss.
        """
        router_probs, router_logits = self._compute_router_probabilities(hidden_states)

        expert_index = torch.argmax(router_probs, dim=-1)
        # expert_index = torch.nn.functional.one_hot(expert_index, num_classes=self.num_experts)

        # Mask tokens outside expert capacity. Sum over each sequence
        # token_priority = torch.cumsum(expert_index, dim=-2)
        # mask if the token routed to to the expert will overflow
        # expert_capacity_mask = token_priority <= self.expert_capacity
        # expert_index = expert_index * expert_capacity_mask

        # router_probs = torch.max(router_probs, dim=-1).values.unsqueeze(-1)
        return expert_index, router_probs, router_logits


# Copied from transformers.models.t5.modeling_t5.T5LayerNorm with T5->SwitchTransformers
class SwitchTransformersLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Construct a layernorm module in the SwitchTransformers style. No bias and no subtraction of mean.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):

        # SwitchTransformers uses a layer_norm which only scales and doesn't shift, which is also known as Root Mean
        # Square Layer Normalization https://arxiv.org/abs/1910.07467 thus varience is calculated
        # w/o mean and there is no bias. Additionally we want to make sure that the accumulation for
        # half-precision inputs is done in fp32

        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # convert into half-precision if necessary
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states


# Copied from transformers.models.t5.modeling_t5.T5DenseActDense with T5->SwitchTransformers




class TaskMoE(torch.nn.Module):
    def __init__(self,
                 hidden_size,
                 expert,
                 num_experts=1,
                 attr_len=3, 
                 expert_capacity=1., 
                 router_bias=True, 
                 router_noise=1e-2,
                 is_scale_prob=True,
                 drop_tokens=True):
        """Initialize an MoE layer.
        Arguments:
            hidden_size (int): the hidden dimension of the model, importantly this is also the input and output dimension.
            expert (torch.nn.Module): the torch module that defines the expert (e.g., MLP, torch.linear).
            num_experts (int, optional): default=1, the total number of experts per layer.
            k (int, optional): default=1, top-k gating value, only supports k=1 or k=2.
            capacity_factor (float, optional): default=1.0, the capacity of the expert at training time.
            eval_capacity_factor (float, optional): default=1.0, the capacity of the expert at eval time.
            min_capacity (int, optional): default=4, the minimum capacity per expert regardless of the capacity_factor.
            noisy_gate_policy (str, optional): default=None, noisy gate policy, valid options are 'Jitter', 'RSample' or 'None'.
            drop_tokens (bool, optional): default=True, whether to drop tokens - (setting to False is equivalent to infinite capacity).
            use_rts (bool, optional): default=True, whether to use Random Token Selection.
            use_tutel (bool, optional): default=False, whether to use Tutel optimizations (if installed).
        """

        super().__init__()


        self.num_experts = num_experts

        # if isinstance(expert, nn.Linear):
        #     self.expert_type = 'linear'
        # elif isinstance(expert, nn.MultiheadAttention):
        #     self.expert_type = 'attention'
        # else:
        #     raise NotImplementedError('please check expert type')

        experts = FusedExperts(expert, expert_capacity, is_scale_prob, num_experts, drop_tokens)

        self.gate = Top1Router(attr_len, 
                            hidden_size, 
                            num_experts, 
                            router_bias=router_bias, 
                            router_noise=router_noise)


        self.experts = experts



    def forward(self, input_data, attr=None, gate_decision=None, **kwargs):
        """ MoE forward
        Arguments:
            attr(Tensor): input to the gate
            input_data (Tensor): input to the layer
        Returns:
            A tuple including output
            * output (Tensor): output of the model
        """

        if attr is None:
            attr = input_data

        if  gate_decision is not None:
            expert_index, router_probs, router_logits = gate_decision
        else:
            expert_index, router_probs, router_logits = self.gate(attr)

        expert_output = self.experts(input_data, router_probs)

        return expert_output, [expert_index, router_probs, router_logits]




class FusedExperts(torch.nn.Module):
    def __init__(self, expert, capacity_factor, is_scale_prob, n_experts, drop_tokens):
        super(FusedExperts, self).__init__()

        self.capacity_factor = capacity_factor
        self.is_scale_prob = is_scale_prob
        self.n_experts = n_experts
        self.drop_tokens = drop_tokens

        self.deepspeed_experts = torch.nn.ModuleList(
            [copy.deepcopy(expert) for i in range(n_experts)])

        # self.bias_merge = self.deepspeed_experts[0].bias is not None

    # def top1_expert_forward(self, x, index1):
    #     return F.linear(
    #         x,
    #         self.deepspeed_experts[index1].weight,
    #         bias=self.deepspeed_experts[index1].bias if self.bias_merge else None,
    #     )


        
    
    def forward(self, x, router_probs):

        batch_size, H, W, d_model = x.shape
        # batch_size, seq_len, d_model = x.shape
        # Flatten the sequence and batch dimensions
        x = x.view(-1, d_model)

        route_prob_max, routes = torch.max(router_probs, dim=-1)

        # Get indexes of tokens going to each expert
        indexes_list = [torch.eq(routes, i).nonzero(as_tuple=True)[0] for i in range(self.n_experts)]

        # Initialize an empty tensor to store outputs
        final_output = x.new_zeros(x.shape)

        # Capacity of each expert.
        # $$\mathrm{expert\;capacity} =
        # \frac{\mathrm{tokens\;per\;batch}}{\mathrm{number\;of\;experts}}
        # \times \mathrm{capacity\;factor}$$
        capacity = int(self.capacity_factor * len(x) / self.n_experts)
        # Number of tokens routed to each expert.
        counts = x.new_tensor([len(indexes_list[i]) for i in range(self.n_experts)])

        # Initialize an empty list of dropped tokens
        dropped = []
        # Only drop tokens if `drop_tokens` is `True`.
        if self.drop_tokens and self.training:
            # Drop tokens in each of the experts
            for i in range(self.n_experts):
                # Ignore if the expert is not over capacity
                if len(indexes_list[i]) <= capacity:
                    continue
                # Shuffle indexes before dropping
                indexes_list[i] = indexes_list[i][torch.randperm(len(indexes_list[i]))]
                # Collect the tokens over capacity as dropped tokens
                dropped.append(indexes_list[i][capacity:])
                # Keep only the tokens upto the capacity of the expert
                indexes_list[i] = indexes_list[i][:capacity]

        # Get outputs of the expert FFNs
        # expert_output = [self.experts[i](x[indexes_list[i], :]) for i in range(self.n_experts)]
        expert_output = [self.deepspeed_experts[i](x[indexes_list[i], :]) for i in range(self.n_experts)]

        # Assign to final output
        for i in range(self.n_experts):
            final_output[indexes_list[i], :] = expert_output[i]

        # Pass through the dropped tokens
        if dropped:
            dropped = torch.cat(dropped)
            final_output[dropped, :] = x[dropped, :]

        if self.is_scale_prob:
            # Multiply by the expert outputs by the probabilities $y = p_i(x) E_i(x)$
            final_output = final_output * route_prob_max.view(-1, 1)
        else:
            # Don't scale the values but multiply by $\frac{p}{\hat{p}} = 1$ so that the gradients flow
            # (this is something we experimented with).
            final_output = final_output * (route_prob_max / route_prob_max.detach()).view(-1, 1)

        # Change the shape of the final output back to `[seq_len, batch_size, d_model]`
        final_output = final_output.view(batch_size, H, W, d_model)

        # Return
        #
        # * the final output
        # * number of tokens routed to each expert
        # * sum of probabilities for each expert
        # * number of tokens dropped.
        # * routing probabilities of the selected experts
        #
        # These are used for the load balancing loss and logging
        return final_output

