import torch

class MLP:
    def __init__(
        self,
        linear_1_in_features,
        linear_1_out_features,
        f_function,
        linear_2_in_features,
        linear_2_out_features,
        g_function
    ):
        """
        Args:
            linear_1_in_features: the in features of first linear layer
            linear_1_out_features: the out features of first linear layer
            linear_2_in_features: the in features of second linear layer
            linear_2_out_features: the out features of second linear layer
            f_function: string for the f function: relu | sigmoid | identity
            g_function: string for the g function: relu | sigmoid | identity
        """
        self.f_function = f_function
        self.g_function = g_function

        self.parameters = dict(
            W1 = torch.randn(linear_1_out_features, linear_1_in_features),
            b1 = torch.randn(linear_1_out_features),
            W2 = torch.randn(linear_2_out_features, linear_2_in_features),
            b2 = torch.randn(linear_2_out_features),
        )
        self.grads = dict(
            dJdW1 = torch.zeros(linear_1_out_features, linear_1_in_features),
            dJdb1 = torch.zeros(linear_1_out_features),
            dJdW2 = torch.zeros(linear_2_out_features, linear_2_in_features),
            dJdb2 = torch.zeros(linear_2_out_features),
        )

        # put all the cache value you need in self.cache
        self.cache = dict()

    def activation(self, function, value):

        if function == "relu":
            return torch.relu(value)
        elif function == "sigmoid":
            return torch.sigmoid(value)
        else:
            return value

    def d_activation(self, function, input, output):

        if function == "relu":
            return (input > 0).type(int)
        elif function == "sigmoid":
            return output * (1-output)
        else:
            return torch.ones(input.shape[0])


    def forward(self, x):
        """
        Args:
            x: tensor shape (batch_size, linear_1_in_features)
        """
       
        # z0 = torch.transpose(x, 0, 1)
        # s1 = torch.mm(self.parameters['W1'], z0) + self.parameters['b1'][:, None]
        # z1 = self.activation(self.f_function, s1)
        
        # s2 = torch.mm(self.parameters['W2'], z1) + self.parameters['b2'][:, None]

        z1 = torch.mm(x, self.parameters['W1'].T) + self.parameters['b1']
        z2 = self.activation(self.f_function, z1)

        z3 = torch.mm(z2, self.parameters['W2'].T) + self.parameters['b2']
        yhat = self.activation(self.g_function, z3)

        self.cache["X"] = x
        self.cache["z1"] = z1 
        self.cache["z2"] = z2 
        self.cache["z3"] = z3 
        self.cache["yhat"] = yhat

        return yhat

    
    def backward(self, dJdy_hat):
        """
        Args:
            dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
        """
        
        dyhat_dz3 = self.d_activation(self.cache['z3'], self.cache['yhat'])

        dJ_dz3 = dJdy_hat * dyhat_dz3
        dJ_dW2 = dJ_dz3 * self.cache["z2"]



    
    def clear_grad_and_cache(self):
        for grad in self.grads:
            self.grads[grad].zero_()
        self.cache = dict()

def mse_loss(y, y_hat):
    """
    Args:
        y: the label tensor (batch_size, linear_2_out_features)
        y_hat: the prediction tensor (batch_size, linear_2_out_features)

    Return:
        J: scalar of loss
        dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
    """
    loss = torch.pow(y-y_hat, 2.0) / y.shape[0]
    dJdy_hat = (2.0 * (y_hat - y)) / y.shape[0]

    return loss, dJdy_hat

def bce_loss(y, y_hat):
    """
    Args:
        y_hat: the prediction tensor
        y: the label tensor
        
    Return:
        loss: scalar of loss
        dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
    """
    loss = -torch.sum(y * torch.log(y_hat) - (1-y) * torch.log(1-y_hat))
    dJdy_hat = -y/y_hat + (1.0-y)/(1.0-y_hat)

    return loss, dJdy_hat



if __name__ == '__main__': 
    net = MLP(
        linear_1_in_features=2,
        linear_1_out_features=20,
        f_function='relu',
        linear_2_in_features=20,
        linear_2_out_features=5,
        g_function='identity'
    )
    x = torch.randn(10, 2)
    y = torch.randn(10, 5)

    net.clear_grad_and_cache()
    y_hat = net.forward(x)

    print(y_hat)









