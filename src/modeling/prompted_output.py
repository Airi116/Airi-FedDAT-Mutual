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

    # text_embeds = torch.cat([question_