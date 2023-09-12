import torch

def albef_prompted_forward(self, image, question, answer=None, alpha=0, k=None, weights=None, train=True):
    image_embeds = self.visual_encoder(image)
    input_tokens_vis = self.prompt_tokens_vis.unsqueeze(0).expand(image_embeds.shape[0], -1).to(image_embeds.device)
    prompt_prompt_vis = self.prompt_embedding_vis(input_tokens_vis) # (B, 5, 768)
    image_embeds = torch.cat([image_embeds[:, :1, :],
                            prompt_prompt_vis,
                            image_embeds[:, 1:, :],], dim=1)
    image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

    # input_tokens_text = self.prompt_tokens_text.unsqueeze(0).expand(question_output.last_hidden_state.shape[0], -1).to(question_output.last_hidden_state.device)
    # prompt_prompt_text = self.prompt_embedding_text(input_tokens_text) # (B, 5, 768)
    # prompt_mask_text = torch.ones(prompt_prompt_text.shape[:2], dtype=torch.long).to(text_embeds.device)
    # question.attention_mask = torch.cat([question.attention_mask[:, :1],
    #                             prompt_mask_text,
    #                             question.attention_mask[:, 1:]], dim=1)
    question_output = self.text_encoder(question.input_ids,  # tokenized question
                                        attention_mask=question.attention_mask,
                                        encoder_hidden_states=image_embeds,
                                        encoder_attention_mask=image_atts,
                                        return_dict=True)  # last_hidden_state: (batch, words, 768)

    # text_embeds = torch.cat([question_output.last_hidden_state[:, :1, :],
    #                         prompt_prompt_text,
    #                         question_output.last_hidden_state[:, 1:, :],], dim=1)
    if train:
        """
        k: number of answers for each question
        weights: weight for each answer
        """
        answer_targets = answer.input_ids.masked_fill(answer.input_ids == self.tokenizer.pad_token_id, -100)

        question_states = []
        question_atts = []
        for b, n in enumerate(k):
            question_states += [question_output.last_hidden_state[b]] * n
            question_atts += [question.attention_mask[b]] * n
        question_states = torch.stack(question_states, 0)
        question_atts = torch.stack(question_atts, 0)

        answer_output = self.text_decoder(answer.input_ids,
                                            attention_mask=answer.attention_mask,
                                            encoder_hidden_states=question_states,
                                            encoder_attention_mask=question_atts,
                                            labels=answer_targets,
                                            return_dict=True,
                                            reduction="none",
                                            )
        loss = weights * answer_output.loss
        loss = loss.sum() / image.size(0)

        return (loss, answer_output.logits[:, :-1, :].contiguous())  # logits: (batch, words, vocab_size(30522))

    else:
        topk_ids, topk_probs = self.rank_answer(question_output.last_hidden_state, question.attention_mask,
                                                answer.input_ids, answer.attention_mask, k)  # answer.input_ids: [num_answers, max_len]; k=128
        return topk_ids, topk_probs

def BERTEmbeddings_prompted_forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        is_decoder=False,
        mode='multi_modal',
):
    r"""
    encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
        Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
        the model is configured as a decoder.
    encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
        Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
        the cross-attention if the model is configured as a decoder. Mask values selected in ``[0, 1]``:
        - 1 for tokens that are **not masked**,
        - 0 for tokens that are **masked**.
    past_key_values (:obj:`tuple(tuple(torch.FloatTensor))` of length :obj:`config.n_layers` with each tuple having 4 tensors of shape :obj:`(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
        Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.
        If :obj:`past_key_values` are used, the user can optionally input only the last :obj:`decoder_input_ids`
        (those that don't have their past key value states given to this model) of shape :obj:`(batch_size, 1)`
        instead of all :obj:`decoder_input_ids` of shape :obj:`(batch_size, sequence_length)`.
    use_cache (:obj:`bool`, `optional`):
        If set to :obj:`True`, :obj:`past_key_values` key value states are returned and can be used to speed up
        decoding (see :obj:`past_key_values`).
    """
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions  # False
    output_hidden_states = (  # False
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict  # True

    if is_decoder:
        use_cache = use_cache if use_cache is not None else self.config.use_cache
    else:
        use_cache = False

    if input_ids is not None and inputs_embeds is not None:
        raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
    elif input_ids is not None:
        input_shape = input_ids.size()
        batch_size, seq_length = input_shape  # question
        device = input_ids.device
    elif inputs_embeds is not None:
        input_shape = inputs_embeds.size()[:-1]
        batch_size, seq_length = input_shape
        device = inputs_embeds.device
    elif encoder_embeds is not None:
        input_shape = encoder_embeds.size()[:-1]
        batch_size, seq_length = input_shape
        device = encoder_embeds.device
    else:
        raise ValueError("You have to specify either input_ids or inputs_embeds or encoder_embeds")

    # past_key_values_length
    past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0  # 0

    if attention_mask is None:
        attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)
    if token_type_ids is None:
        toke