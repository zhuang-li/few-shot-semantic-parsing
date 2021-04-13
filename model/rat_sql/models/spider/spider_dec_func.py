import torch


def compute_align_loss(desc_enc):
    '''model: a nl2code decoder'''
    # find relevant columns

    mc_att_on_rel_col = desc_enc.m2c_align_mat
    #print (mc_att_on_rel_col.size())
    mc_max_rel_att, _ = mc_att_on_rel_col.max(dim=0)
    mc_max_rel_att.clamp_(min=1e-9)


    align_loss = - torch.log(mc_max_rel_att).mean()
    return align_loss


def compute_pointer_with_align(
        model,
        node_type,
        prev_state,
        prev_action_emb,
        parent_h,
        parent_action_emb,
        desc_enc):
    new_state, attention_weights = model._update_state(
        node_type, prev_state, prev_action_emb, parent_h,
        parent_action_emb, desc_enc)
    # output shape: batch (=1) x emb_size
    output = new_state[0]
    memory_pointer_logits = model.pointers[node_type](
        output, desc_enc.memory)
    memory_pointer_probs = torch.nn.functional.softmax(
        memory_pointer_logits, dim=1)
    # pointer_logits shape: batch (=1) x num choices
    if node_type == "column":
        pointer_probs = torch.mm(memory_pointer_probs, desc_enc.m2c_align_mat)
    else:
        assert node_type == "table"
        pointer_probs = torch.mm(memory_pointer_probs, desc_enc.m2t_align_mat)
    pointer_probs = pointer_probs.clamp(min=1e-9)
    pointer_logits = torch.log(pointer_probs)
    return output, new_state, pointer_logits, attention_weights
