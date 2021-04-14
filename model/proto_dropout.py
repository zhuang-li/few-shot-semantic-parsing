import torch.nn as nn

class ProtoDropout(nn.Module):
    """ LockedDropout applies the same dropout mask to every time step.

    **Thank you** to Sales Force for their initial implementation of :class:`WeightDrop`. Here is
    their `License
    <https://github.com/salesforce/awd-lstm-lm/blob/master/LICENSE>`__.

    Args:
        p (float): Probability of an element in the dropout mask to be zeroed.
    """

    def __init__(self, p=0.5):
        self.p = p
        super().__init__()

    def forward(self, x):
        """
        Args:
            x (:class:`torch.FloatTensor` [type length, embedding size]): Input to
                apply dropout too.
        """
        x = x.clone()
        mask = x.new_empty(1, x.size(0), requires_grad=False).bernoulli_(1 - self.p)
        #print (x.size())
        #print (mask.size())
        return (x.t() * mask).t(), mask


    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'p=' + str(self.p) + ')'