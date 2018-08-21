import torch
import HighDimFilter

def high_dim_filter_cpu(input, rgb, bilateral, theta_alpha, theta_beta, theta_gamma, backwards):
    '''
    :input: [height, width, channel]
    :return:
    '''
    assert input.is_cuda == False, 'Only cpu tensor supported'
    assert rgb.is_cuda == False, 'Only cpu tensor supported'
    H, W, C = input.shape
    output = torch.zeros(H, W, C)
    HighDimFilter.filter(output, input, rgb, bilateral, theta_alpha, theta_beta, theta_gamma, backwards)

    return output

class HighDimFilterFunction(torch.autograd.Function):
    # Note that both forward and backward are @staticmethods
    @staticmethod
    def forward(ctx, input, rgb, bilateral, theta_alpha, theta_beta, theta_gamma):
        assert input.shape[:2] == rgb.shape[:2], 'input and rgb should have same height and width'
        input = input.cpu()
        rgb = rgb.cpu()
        ctx.save_for_backward(input, rgb)
        ctx.params = [bilateral, theta_alpha, theta_beta, theta_gamma]

        output = high_dim_filter_cpu(input, rgb, bilateral, theta_alpha, theta_beta, theta_gamma, False)
        output = output.cuda()
        return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        input, rgb = ctx.saved_tensors
        bilateral, theta_alpha, theta_beta, theta_gamma = ctx.params
        grad_input = grad_rgb = None

        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.

        grad_output = grad_output.cpu()

        if ctx.needs_input_grad[0]:
            grad_input = high_dim_filter_cpu(grad_output, rgb, bilateral, theta_alpha, theta_beta, theta_gamma, True)
        if ctx.needs_input_grad[1]:
            grad_rgb = torch.zeros(rgb.shape)
        grad_input = grad_input.cuda()

        return grad_input, grad_rgb, None, None, None, None

if __name__ == '__main__':
    from torch.autograd import gradcheck, Variable

    # gradcheck takes a tuple of tensors as input, check if your gradient
    # evaluated with these tensors are close enough to numerical
    # approximations and returns True if they all verify this condition.
    input = (Variable(torch.randn(10, 10, 5).float().cuda(), requires_grad=True),
             Variable(torch.randn(10, 10, 5).float().cuda(), requires_grad=False),
             True, 1000., 1000., 1000.)
    test = gradcheck(HighDimFilterFunction.apply, input, eps=1e-5, atol=2e-3)
    print(test)
