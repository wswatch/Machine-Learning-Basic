import numpy as np

#
#
# The following code uses the following concepts/functions from NumPy, and it
# may help to brush up on them from the documentation.
#
# - Broadcasting
# - np.pad
# - np.reshape
# - np.tensordot (optional)
# - np.newaxis
# - np.moveaxis

DEBUG = True

class relu:
    
    def function(self, X):
        '''Return elementwise relu values'''
        return(np.maximum(np.zeros(X.shape), X))
    
    def derivative(self, X):
        # An example of broadcasting
        return((X >= 0).astype(int))
        
class no_act:
    """Implement a no activation function and derivative"""
    
    def function(self, X):
        return(X)
    
    def derivative(self, X):
        return(np.ones(X.shape))

# Set of allowed/implemented activation functions
ACTIVATIONS = {'relu': relu,
               'no_act': no_act}    
    
class CNNLayer:
    """
    Implement a class that processes a single CNN layer.
    
    Let i be the index on the neurons in the ith layer, and j be the index on the 
    nuerons in the next outler layer.  (Following Russell-Norvig notation.) Implement 
    the following:
    
    0. __init__: Initalize filters.
    
    1. forward step: Input a_i values.  Output a_j values.  Make copies of a_i values 
       and in_j values since needed in backward_step and filter_gradient.
    
    2. backward_step: Input (del L)/(del a_j) values.  Output (del L)/(del a_i).
    
    3. filter_gradient: Input (del L)/(del a_j) values. Output (del L)/(del w_{ij}) values.
    
    4. update: Given learning rate, update filter weights. 
    
    """
    
    def __init__(self, n, filter_shape, activation='no_act', stride = 1):
        """
        Initialize filters.
        
        filter_shape is (width of filter, height of filter, depth of filter). Depth 
        of filter should match depth of the forward_step input X.
        """
        
        self.num_filters = n
        self.stride = stride
        self.filter_shape = filter_shape
        self.act = activation
        try:
            self.filter_height = filter_shape[0]
            self.filter_width = filter_shape[1]
            self.filter_depth = filter_shape[2]
        except:
            raise Exception(f'Unexpected filter shape {filter_shape}')
        try:
            # Create an object of the activation class
            self.activation = ACTIVATIONS[activation]() 
        except:
            raise Exception(f'Unknown activation: {activation}')
        self.filters = self.filters_init()
        self.biases = self.biases_init()
        self.num_examples = None 
        # Set num_of_examples during forward step, and use to verify
        # consistency during backward step.  Similarly the data height, 
        # width, and depth.
        self.data_height = None
        self.data_width = None
        self.data_depth = None
        self.data_with_pads = None
        self.in_j = None  # the in_j values for next layer.
        
    def filters_init(self):
        return np.random.random((self.num_filters, self.filter_height,
                                 self.filter_width, self.filter_depth))
    
    def biases_init(self):
        return np.random.random(self.num_filters)
    
    def set_filters(self, filters, biases):
        """Set filters to given weights.
        
           Useful in debugging."""
        if filters.shape != (self.num_filters, self.filter_height,
                                 self.filter_width, self.filter_depth):
            raise Exception(f'Mismatched filter shapes: stored '
                            f'{self.num_filters} {self.filter_shape} vs '
                            f'{filters.shape}.')
        if biases.shape != (self.num_filters,):
            raise Exception((f'Mismatched biases: stored '
                             f'{self.num_filters} vs '
                             f'{biases.shape}.'))
        self.filters = filters.copy()
        self.biases = biases.copy()
        
    def forward_step(self, X, pad_height=0, pad_width=0):
        """
        Implement a forward step.
        
        X.shape is (number of examples, height of input, width of input, depth of input).
        """
        
        try:
            # Store shape values to verify consistency during backward step
            self.num_examples = X.shape[0]
            self.data_height = X.shape[1]
            self.data_width = X.shape[2]
            self.data_depth = X.shape[3]
        except:
            raise Exception(f'Unexpected data shape {X.shape}')
        if self.data_depth != self.filter_depth:
            raise Exception(f'Depth mismatch: filter depth {self.filter_depth}'
                            f' data depth {self.data_depth}')
        self.pad_height = pad_height
        self.pad_width = pad_width
        self.input_height = self.data_height + 2 * self.pad_height
        self.input_width = self.data_width + 2 * self.pad_width
        
        # Add pad to X.  Only add pads to the 1, 2 (ht, width) axes of X, 
        # not to the 0, 4 (num examples, depth) axes.
        # 'constant' implies 0 is added as pad.
        X = np.pad(X, ((0,0),(pad_height, pad_height), 
                      (pad_width, pad_width), (0,0)), 'constant')
        
        # Save this for the update step
        # self.data_with_pads = X.copy() #REMOVE THIS
        # Save a copy for computing filter_gradient
        self.a_i = X.copy()  #
        
        # Get height, width after padding
        height = X.shape[1]
        width = X.shape[2]

        # Don't include pad in formula because height includes it.
        output_height = ((height - self.filter_height)/self.stride + 1)
        output_width = ((width - self.filter_width)/self.stride + 1)    
        if (
            output_height != int(output_height) or 
            output_width != int(output_width)
        ):
            raise Exception(f"Filter doesn't fit: {output_height} x {output_width}")
        else:
            output_height = int(output_height)
            output_width = int(output_width)
            
        #####################################################################
        # There are two ways to convolve the filters with X.
        # 1. Using the im2col method described in Stanford 231 notes.
        # 2. Using NumPy's tensordot method.
        #
        # (1) requires more code.  (2) requires understanding how tensordot
        # works.  Most likely tensordot is more efficient.  To illustrate both,
        # in the code below data_tensor is constructed using (1) and 
        # new_data_tensor is constructed using (2).  You may use either.
            
        # Stanford's im2col method    
        # Construct filter tensor and add biases
        filter_tensor = self.filters.reshape(self.num_filters, -1)
        filter_tensor = np.hstack((self.biases.reshape((-1,1)), filter_tensor))
        # Construct the data tensor
        # The im2col_length does not include the bias terms
        # Biases are later added to both data and filter tensors
        im2col_length = self.filter_height * self.filter_width * self.filter_depth
        num_outputs = output_height * output_width
        data_tensor = np.empty((self.num_examples, num_outputs, im2col_length))
        for h in range(output_height):
            for w in range(output_width):
                hs = h * self.stride
                ws = w * self.stride
                data_tensor[:,h*output_width + w, :] = X[:,hs:hs+self.filter_height,
                                    ws:ws+self.filter_width,:].reshape(
                                        (self.num_examples,-1))  
        # add bias-coeffs to data tensor
        data_tensor = np.concatenate((np.ones((self.num_examples, num_outputs, 1)),
                                 data_tensor), axis=2)
        output_tensor = np.tensordot(data_tensor, filter_tensor, axes=([2],[1]))
        output_tensor = output_tensor.reshape(
            (self.num_examples,output_height,output_width,self.num_filters))
        
        
        
        # NumPy's tensordot based method
        new_output_tensor = np.empty((self.num_examples, output_height, 
                                      output_width, self.num_filters))
        for h in range(output_height):
            for w in range(output_width):
                hs = h * self.stride
                ws = w * self.stride
                new_output_tensor[:,h,w,:] = np.tensordot(
                                                X[:, # example
                                                  hs:hs+self.filter_height, # height
                                                  ws:ws+self.filter_width,  # width
                                                  : # depth
                                                ], 
                                                self.filters[:, # filter 
                                                             :, # height
                                                             :, # width
                                                             :  # depth
                                                ], 
                                                axes = ((1,2,3),(1,2,3))
                                              )
                # Add bias term
                new_output_tensor[:,h,w,:] = (new_output_tensor[:,h,w,:] + 
                                              self.biases)
        # Check both methods give the same answer
        assert np.array_equal(output_tensor, new_output_tensor)
                
        
        self.in_j = output_tensor.copy() # Used in backward_step.
        output_tensor = self.activation.function(output_tensor) # a_j values
        return(output_tensor)
# do 2D gradient      
    def backward_step(self, D):
        """
        Implement the backward step and return (del L)/(del a_i). 
        
        Given D=(del L)/(del a_j) values return (del L)/(del a_i) values.  
        D (delta) is of shape (number of examples, height of output (i.e., the 
        a_j values), width of output, depth of output)"""
                
        try:
            num_examples = D.shape[0]
            delta_height = D.shape[1]
            delta_width = D.shape[2]
            delta_depth = D.shape[3]
        except:
            raise Exception(f'Unexpected delta shape {D.shape}')
        if num_examples != self.num_examples:
            raise Exception(f'Number of examples changed from forward step: '
                             f'{self.num_examples} vs {num_examples}')
        if delta_depth != self.num_filters:
            raise Exception(f'Depth mismatch: number of filters {self.num_filters}' 
                            f' delta depth {delta_depth}')
        # Make a copy so that we can change it
        prev_delta = D.copy()
        if prev_delta.ndim != 4:
            raise Exception(f'Unexpected number of dimensions {D.ndim}')
        new_delta = None
        
        ####################################################################
        D = delta_after(D, self.in_j, self.act)
        # WRITE YOUR CODE HERE
        F = self.filters
        new_delta = np.zeros(self.a_i.shape)
        depth = F.shape[3]
        
        for i in range(num_examples):
            for j in range(depth):
                for k in range(delta_depth):
                    new_delta[i,:,:,j] += part_gradient(D[i,:,:,k], F[k,:,:,j], new_delta.shape[1], new_delta.shape[2], self.stride)
        return(new_delta)
# let rotation 180% Delta(filter) and input do convolution   
    def filter_gradient(self, D):
        """
        Return the filter_gradient.
        
        D = (del L)/(del a_j) has shape (num_examples, height, width, depth=num_filters)
        The filter_gradient (del L)/(del w_{ij}) has shape (num_filters, filter_height, 
        filter_width, filter_depth=input_depth)
        
        """
         
        if DEBUG and D.ndim != 4:
            raise Exception(f'D has {D.ndim} dimensions instead of 4.')
        # D depth should match number of filters
        D_depth = D.shape[3]
        if DEBUG:
            if D_depth != self.num_filters:
                raise Exception(f'D depth {D_depth} != num_filters'
                                f' {self.num_filters}')
            if D.shape[0] != self.num_examples:
                raise Exception(f'D num_examples {D.shape[0]} !='
                                f'num_examples {self.num_examples}')
        f_gradient = None
        
        ####################################################################
        # WRITE YOUR CODE HERE
        #print(self.a_i)
        #print('***************************************')
        D = delta_after(D, self.in_j, self.act)
        F = self.filters
        num_filter, filter_height, filter_width, filter_depth = F.shape
        f_gradient = np.zeros([self.num_filters, filter_height, filter_width, filter_depth])
        data = self.a_i
        for i in range(self.num_filters):
            for j in range(filter_depth):
                for k in range(self.num_examples):
                    d = D[k,:,:,i]
                    rd = np.rot90(d)
                    rd = np.rot90(rd)
                    f_gradient[i,:,:,j] += convolution(data[k,:,:,j], d, self.stride)
        return(f_gradient)
# 2D gradient(h,w)
def part_gradient(D, W, h, w, stride):
    ni,nj = D.shape
    nu,nv = W.shape
    res = np.zeros([h, w])
    for i in range(ni):
        for j in range(nj):
            for u in range(nu):
                for v in range(nv):
                    res[i*stride+u][j*stride+v] += D[i][j]*W[u][v]
    return res
# 2D convolution
def convolution(X, F, stride=1):
    xi,xj = X.shape
    fi,fj = F.shape
    h = int((xi - fi)/stride) + 1
    w = int((xj - fj)/stride) + 1
    res = np.zeros([h,w])
    for i in range(h):
        for j in range(w):
            si = i*stride
            sj = j*stride
            res[i][j] = np.sum(X[si: si+fi, sj: sj+fj] * F)
    return res 
# if relu, then if output is less than 0, then the delta will be 0 (dL/d in_j)
def delta_after(data, o, activation):
    res = data.copy()
    if activation == 'relu':
        a,b,c,d = data.shape
        for i in range(a):
            for j in range(b):
                for k in range(c):
                    for m in range(d):
                        if o[i][j][k][m] <= 0:
                            res[i][j][k][m] = 0
    return res

# testing
def test_CNNLayer_cs231():
    """
    Longer test example based on demo at Stanford's CS231
    that has depth > 1, stride = 2, but only tests forward steps.
    """

    W0 = np.empty((3,3,3), float)
    W1 = np.empty((3,3,3), float)
    X = np.empty((7,7,3), float)
    X_no_pad = np.empty((5,5,3), float)

    # filter W0
    W0[:, :, 0] = np.array([
            [-1, -1, 0],
            [-1, 0, 1],
            [-1, 1, 1]])
    W0[:, :, 1] = np.array([
            [-1, -1, 0],
            [0, 0, 0],
            [0, 1, 1]])
    W0[:, :, 2] = np.array([
            [1, -1, 1],
            [0, -1, 1],
            [0, -1, 0]])
    b0 = 1
        
    # filter W1    
    W1[:, :, 0] = np.array([
            [0, 0, 0],
            [-1, 1, 1],
            [1, 1, 1]])
    W1[:, :, 1] = np.array([
            [0, -1, 1],
            [-1, 0, 1],
            [0, -1, 1]])
    W1[:, :, 2] = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [-1, 0, -1]])
    b1 = 0    
        
    X[:,:,0] = np.array([
            [0] * 7,
            [0, 0, 0 ,0, 2, 1, 0],
            [0, 0, 0 ,0, 2, 2, 0],
            [0, 1, 2 ,1, 2, 1, 0],
            [0, 2, 0 ,0, 1, 0, 0],
            [0, 1, 1 ,0, 0, 1, 0],
            [0] * 7])
    X[:,:,1] = np.array([
            [0] * 7,
            [0, 0, 0 ,1, 1, 0, 0],
            [0, 1, 2 ,0, 1, 0, 0],
            [0, 2, 0 ,2, 2, 0, 0],
            [0, 0, 1 ,1, 0, 0, 0],
            [0, 2, 2 ,2, 2, 1, 0],
            [0] * 7])
    X[:,:,2] = np.array([
            [0] * 7,
            [0, 1, 2 ,1, 0, 0, 0],
            [0, 0, 1 ,2, 1, 2, 0],
            [0, 2, 2 ,1, 0, 2, 0],
            [0, 0, 2 ,1, 2, 0, 0],
            [0, 0, 1 ,1, 1, 0, 0],
            [0] * 7])
    
    X_no_pad[:,:,0] = np.array([
            [0, 0, 0, 2, 1],
            [0, 0, 0, 2, 2],
            [1, 2, 1, 2, 1],
            [2, 0, 0, 1, 0],
            [1, 1, 0, 0, 1]])
    X_no_pad[:,:,1] = np.array([
            [0, 0 ,1, 1, 0],
            [1, 2 ,0, 1, 0],
            [2, 0 ,2, 2, 0],
            [0, 1 ,1, 0, 0],
            [2, 2 ,2, 2, 1]])
    X_no_pad[:,:,2] = np.array([
            [1, 2 ,1, 0, 0],
            [0, 1 ,2, 1, 2],
            [2, 2 ,1, 0, 2],
            [0, 2 ,1, 2, 0],
            [0, 1 ,1, 1, 0]]) 
    
    output_relu = np.array(
[[[[ 5.,  0.],
   [ 3.,  6.],
   [ 0.,  1.]],
  [[ 6.,  5.],
   [ 0.,  2.],
   [ 0.,  0.]],
  [[ 3.,  5.],
   [ 1.,  0.],
   [ 2.,  0.]]],
 [[[ 5.,  0.],
   [ 3.,  6.],
   [ 0.,  1.]],
  [[ 6.,  5.],
   [ 0.,  2.],
   [ 0.,  0.]],
  [[ 3.,  5.],
   [ 1.,  0.],
   [ 2.,  0.]]],
 [[[ 5.,  0.],
   [ 3.,  6.],
   [ 0.,  1.]],
  [[ 6.,  5.],
   [ 0.,  2.],
   [ 0.,  0.]],
  [[ 3.,  5.],
   [ 1.,  0.],
   [ 2.,  0.]]]]
    )  
    
    output_no_act = np.array(
  [[[[  5.,   0.],
   [  3.,   6.],
   [ -3.,   1.]],

  [[  6.,   5.],
   [ -1.,   2.],
   [-10.,  -4.]],

  [[  3.,   5.],
   [  1.,  -1.],
   [  2.,   0.]]],


 [[[  5.,   0.],
   [  3.,   6.],
   [ -3.,   1.]],

  [[  6.,   5.],
   [ -1.,   2.],
   [-10.,  -4.]],

  [[  3.,   5.],
   [  1.,  -1.],
   [  2.,   0.]]],


 [[[  5.,   0.],
   [  3.,   6.],
   [ -3.,   1.]],

  [[  6.,   5.],
   [ -1.,   2.],
   [-10.,  -4.]],

  [[  3.,   5.],
   [  1.,  -1.],
   [  2.,   0.]]]]
    )
    
    #Y = np.empty((2, 2, 3))
    #Y[:, :, 0] = np.array([[1, 2], [3, 4]])
    #Y[:, :, 1] = np.array([[5, 6], [7, 8]])
    #Y[:, :, 2] = np.array([[9, 10], [11, 12]])
    #X=Y
                      
    conv = CNNLayer(2, (3, 3, 3), 'no_act', stride=2)
    conv.set_filters(np.stack((W0,W1), axis=0), np.array([b0, b1]))
    output = (conv.forward_step(np.stack((X,X,X), axis=0)))
    assert np.array_equal(output_no_act, output)
    
    conv = CNNLayer(2, (3, 3, 3), 'relu', stride=2)
    conv.set_filters(np.stack((W0,W1), axis=0), np.array([b0, b1]))
    output = (conv.forward_step(np.stack((X,X,X), axis=0)))
    assert np.array_equal(output_relu, output)
    
    conv = CNNLayer(2, (3, 3, 3), 'relu', stride=2)
    conv.set_filters(np.stack((W0,W1), axis=0), np.array([b0, b1]))
    XX = X_no_pad
    output = conv.forward_step(np.stack((XX,XX,XX), axis=0), 1, 1)
    assert np.array_equal(output_relu, output)
    print('finish!')
test_CNNLayer_cs231()    
def test_CNNLayer_small():
    """
    Small test example with depth=1 that tests forward and
    backward steps assuming no activation.
    """
    # X = a_i
    X = np.array([[2,3],[4,1]])
    # Two filters
    F1 = np.array([[1, 0],[0,1]])
    F2 = np.array([[1, 2],[0,1]])
    # Expected a_j values
    output_no_act_F1 = np.array(
        [[[[ 2.],
           [ 3.],
           [ 0.]],
          [[ 4.],
           [ 3.],
           [ 3.]],
          [[ 0.],
           [ 4.],
           [ 1.]]]])
    output_no_act_F2 = np.array(
        [[[[ 2.],
           [ 3.],
           [ 0.]],
          [[ 8.],
           [ 9.],
           [ 3.]],
          [[ 8.],
           [ 6.],
           [ 1.]]]])
    # Initialize cnn layer
    conv = CNNLayer(1, (2,2,1), 'no_act', stride=1)

    # Test forward step with filter F1 and bias 0
    conv.set_filters(F1[np.newaxis,:,:,np.newaxis], np.array([0]))
    X = X[np.newaxis,:,:,np.newaxis]
    output = conv.forward_step(X, 1, 1)
    assert np.array_equal(output, output_no_act_F1)

    # Test both backward and forward steps with filter F2 and bias 0. 
    conv.set_filters(F2[np.newaxis,:,:,np.newaxis], np.array([0]))
    output = conv.forward_step(X, 1, 1)
    assert np.array_equal(output, output_no_act_F2)
    # delta contains (del L)/(del a_j)
    delta = np.array(
    [[6, 0, 3],
     [3, 0, 1],
     [0, 0, 2]])
    # Add axis to make it a 4-d tensor
    DD = delta[np.newaxis,:,:,np.newaxis]
    # DD_prev is expected (del L)/(del a_i)
    DD_prev = np.array(
        [[[  6.],
           [ 12.],
           [  3.],
           [  6.]],
          [[  3.],
           [ 12.],
           [  1.],
           [  5.]],
          [[  0.],
           [  3.],
           [  2.],
           [  5.]],
          [[  0.],
           [  0.],
           [  0.],
           [  2.]]])    
    new_delta = conv.backward_step(DD)
    assert np.array_equal(new_delta, DD_prev[np.newaxis,:,:])
    print('finish!')
test_CNNLayer_small()
def test_no_activation():

# Test example with depth with no activation, testing all three function:
# forward_step, backward_step, and filter gradient

# The notation here is different, it corresponds to the pseudocode.

# It is easier for humans to read tensors in the order
# (example, depth, height, width).  So we write the
# tensors in that order, but convert them to the order
# (example, height, width, depth) expected by the code
# before calling the functions.

    L = np.array(
        [ 
            [ # Example 0
                  [ # Depth 0
                        [1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]
                  ],
                  [ # Depth 1
                        [-1, -1, -1],
                  [2,   2,  2],
                  [0,   1,  3]
                  ]
            ],
            [ # Example 1
                  [ # Depth 0
                        [19, 20, 21],
                  [22, 23, 24],
                  [25, 26, 27]
                  ],
                  [ # Depth 1
                        [28, 29, 30],
                  [31, 32, 33],
                  [34, 35, 36]
                  ]
            ]
        ])

    F = np.array(
        [ 
            [# Filter 0
                 [ # Depth 0
                       [1, -1],
                 [2,  0]
                 ],
                 [ # Depth 1
                       [ 0,  0],
                 [-1, -1]
                 ]
            ],
            [# Filter 1
                 [ # Depth 0
                       [0, -1],
                 [3,  0]
                 ],
                 [ # Depth 1
                       [ 1,  2],
                 [-1, -1]
                 ]
            ]
        ]
        )

    delta_prime = np.array(
        [ 
            [ # Example 0
                  [ # Depth 0 (this corresponds to Filter 0, 
                        #          not L Depth 0)
                  [-1, 1],
                  [ 0, 2]
                  ],
                  [ # Depth 1
                        [0, -1],
                  [1,  0]
                  ]
            ],
            [ # Example 1
                  [ # Depth 0
                       [-2, 0],
                  #[0,0],
                  [ 0, 0]
                  ],
                  [ # Depth 1
                        [0,  1],
                  [0,  0]
                  ]
            ]
    
        ])

    F_delta_depth_first = np.array(
        [
            [ # Filter 0
                  [ # Depth 0 (this corresponds to L Depth 0
                        #          not delta_prime Depth 0)]
                  [-27, -27],
                  [-27, -27]
                  ],
                  [ # Depth 1
                        [-52, -54],
                  [-60, -58]
                  ]
            ],
            [ # Filter 1
                  [ # Depth 0 
                        [22, 23],
                  [25, 26]
                  ],
                  [ # Depth 1
                        [32, 33],
                  [30, 32]
                  ]
            ]
        ])

    L_prime = np.array(
        [[[[  4.,   3.],
               [  6.,   5.]],

              [[ 13.,  21.],
                   [ 12.,  20.]]],


        [[[-19.,  69.],
              [-19.,  72.]],

        [[-19.,  78.],
             [-19.,  81.]]]]
        )

    L_delta = np.array(
        [[[[-1.,  0.],
               [ 2., -1.],
               [ 0., -2.]],

              [[-2.,  2.],
                   [ 0.,  3.],
              [-2.,  0.]],

              [[ 3., -1.],
                   [ 4., -3.],
              [ 0., -2.]]],


        [[[-2.,  0.],
              [ 2.,  1.],
              [-1.,  2.]],

        [[-4.,  2.],
             [ 3.,  1.],
        [ 0., -1.]],

        [[ 0.,  0.],
             [ 0.,  0.],
        [ 0.,  0.]]]]
        )    

    F_delta = np.array(
        [[[[-27., -52.],
               [-27., -54.]],

              [[-27., -60.],
                   [-27., -58.]]],


        [[[ 22.,  32.],
              [ 23.,  33.]],

        [[ 25.,  30.],
             [ 26.,  32.]]]]
        )

    # Convert L from (example, depth, height, width) to
    # (example, height, width, depth)    
    L = np.moveaxis(L,1,3)

    # Convert F from (filter, depth, height, width) to
    # (filter, height, width, depth)
    F = np.moveaxis(F,1,3)

    # Convert delta_prime from (example, depth, height, width) to
    # (example, height, width, depth)
    delta_prime = np.moveaxis(delta_prime, 1, 3)

    conv = CNNLayer(2, (2, 2, 2), activation="no_act")
    conv.set_filters(F, biases = np.array([1, 0]))

    L_prime_out = conv.forward_step(L)
    L_delta_out = conv.backward_step(delta_prime)
    F_delta_out = conv.filter_gradient(delta_prime)    

    assert np.array_equal(L_prime, L_prime_out)
    assert np.array_equal(L_delta, L_delta_out)
    assert np.array_equal(F_delta, F_delta_out)
    print('finish')

test_no_activation()
