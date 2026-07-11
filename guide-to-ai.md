# Guide to AI

## Perceptron

A single-layer perceptron is not able to learn XOR functions. This can be fixed by adding one more
layer however. You can represent what a 2-input variables, 2-neuron + bias "baby perceptron" can
learn in a cartesian space: it learn the division between spaces as a straight line. Its formula is:

ŷ = w1 x1 + w2 x2 + b

Widrow and Hoff invented the LMS algorithm to compute the weights w1, w2 and b based on
desired output y. It computes the rate of change of the squared error, E = (y - ŷ)^2^
(taking the derivative of the error function). This way we can apply a gradient descent.

To improve the perceptron, we need an activation function at each neuron:
- The most basic is the McCulloch-Pitts neuron, which is a step function returning
  either 0 or 1. The problem is that it creates a space with flat surfaces and LMS gets stuck.
- The modern one is a sigmoid function:

f(x) = 1 / 1 + e^-x^

Today LMS is replaced with the backpropagation algorithm.

## GPT 3 (2020)

It was the largest neural network at the time with 175 billion learnable weights.

Those are spread across 96 layers, each made of 2 compute blocks:

- Attention (~50K neurons), a kind of specialized perceptron where the weights are
  controlled by other perceptrons.
- Multilayer perceptron (2 layers, ~12K neurons)

Today GPT 3 fits around 10 GPUs. GPT 4 is reported to be 10 times bigger.

## Gradient descent

We'll use Llama as an example. It only has 16 layers and 1.2 billion parameters. Its
vocabulary consists of 128K tokens. Note that most tokenizers are learned and based on
the most frequent combination of characters found in training texts.

When we send a message "The capital of France is", the text is first transformed into
tokens, each represented by a single number. For each input token (let's pretend there
are only 5), the model returns a prediction of what token will come next, in the form of
a vector of probabilities, one per possible token. So the model returns 5 vectors of
128K values (with one predicted value for each of the 128K possible tokens in the
vocabulary). Our 5th vector has maximum probability of 0.39 for the token "Paris". The
next most likely token is "a", with a lower probability (for sentences like "The capital
of France is a beautiful place to visit").

When we train the model, if we wanted it to predict "Paris" with probability 1 (and all
other tokens with probability 0), the error is:

        Error = 1 - Pi

This measure is called **L1 loss**. P("Paris") = 0.39 => L1 loss = 1 - 0.39 = 0.61.

But we use another more effective error function called **cross-entropy loss**:

        Error = - ln(Pi)

(Here the loss would be 0.94.)

So how do we tune the parameters, for example in the last layer, to maximize the
probability of "Paris"? We can compute the slope of the loss curve, telling us which way
is downhill in each direction. From here we can put the slopes into a single vector
called the gradient. And we do this in iterations: take a small iterative step downhill,
recompute the gradient, etc. This is called the gradient descent.

To avoid getting stuck into local minimums, the approach is to choose a random direction
into our high-dimensional space, and measure our loss as we take small steps into this
direction. Note that here choosing a random direction means generating 1.2 billion
random numbers, one for each model parameter, and taking a small step means multiplying
these 1.2 billion numbers by a small scaling factor and adding these scaled random
numbers to our weights and then recomputing our loss.

## Backpropagation

The algorithm was invented by Werbos in 1970.

Here we will be using a much smaller model, only able to predict the city you are in
based on your GPS coordinates (Madrid, Paris, Berlin). The training data is GPS
coordinates from spots in these cities (e.g. the latitude and longitude degrees of the
Eiffel Tower).


To make it real simple at the beginning, we will only use the longitude. So our model:

- takes a single input x (longitude),
- returns 3 numbers: the probability for each city (ŷ1, ŷ2, ŷ3).

Our model has just 3 neurons, one for each city. Each neuron's job is very simple: it
multiplies its input, h1, by a learnable parameter, the weight w1, and adds another
learnable parameter, the bias b1, and outputs the result. This model is completely
equivalent to a simple linear equation:

        y = mx + b

Here is what our model looks like:

            *m1 → +b1 → = h1    → softmax →   ŷ1 = 0.1  → Madrid
          ↗
        x → *m2 → +b2 → = h2    → softmax →   ŷ2 = 0.7  → Paris
          ↘
            *m3 → +b3 → = h3    → softmax →   ŷ3 = 0.2  → Berlin

Notes:

- Above we call the output of the first neuron h1 instead of ŷ1, because that's before
  softmax.
- Each neuron looks like a line with slope m.
- If a neuron has 2 inputs, the formula becomes `y = m1x1 + m2x2 + b` and now looks like
  a plane.
- A typical neuron in a LLM will have in the order of 10k inputs, and follows the same
  pattern. Geometrically that looks like a hyper-plane in a 10k dimensional space.

Why softmax? The output of each neuron's equation is any positive or negative value, but
what we want is a probability between 0 and 1. To ensure that that happens, and that all
probabilities add up to 1, we use the softmax function. The probability of Madrid is
this function that takes the results of all neurons:

                                  e^h1           e^hi
        ŷ1 = Softmax(h1) = ------------------ = -------
                           e^h1 + e^h2 + e^h3   Σj e^hj

The exponentials in the softmax equation amplify the differences between the neurons'
outputs, assigning more, but not all, probability to the neuron with the maximum output
value. That is why it is called *soft* max. For example:

- If the Madrid, Paris and Berlin neurons output 1, 2, 1, softmax will assign a
  probability of 58% to Paris.
- If the outputs are 1, 10, 1, then softmax assigns a probability of 99.98% to Paris.

Before training, the model's weights are randomly initialized. For example let's set:

- m1 = 1, m2 = 0, m3 = -1,
- all bias values b = 0.

Now we pass a longitude, for example the center of Paris 2.3514°. The positive value for
m1 gives a large h1 value of 2.3514, leading the model to incorrectly predict that we
are in Madrid with probability of 0.91.

                    m1=1.0  → b1=0 → h1=2.35  → softmax → Madrid ŷ1=0.91
                  ↗
        x=2.3514° → m2=0.0  → b2=0 → h2=0.00  → softmax → Paris ŷ2=0.09
                  ↘
                    m3=-1.0 → b3=0 → h3=-2.35 → softmax → Berlin ŷ3=0.00

Now let's use the **cross-entropy loss** to measure performance. The current probability
for Paris is 9%, and the loss for Paris is:

        Loss = -ln(0.09) = 2.45

If our model had predicted Paris at 100%, the loss would be `-ln(1.0) = 0`.

So now our job is to change our 6 m and b parameters to make our loss go down. As we saw
before, we can do this by:

- Compute the slope of each of our parameters relative to the loss,
- Combine these slopes into a vector called the gradient,
- Use the gradient to make iterative updates to the weights.

Let's make it more tangible. Let's explore how the loss varies when we change a single
parameter, m2 (currently 0). If we increase m2 to 0.1 and recompute, we see that this
increases the probability of Paris to 0.107, reducing the loss to 2.24. So now we have 2
computations of our loss for two different values of m2. We have effectively estimated
the value of our slope `𝚫L / 𝚫m2`: if we increase m2, the loss will go down.

        Loss ▲
        2.25 |\
             | \
             |  \             𝚫L / 𝚫m2 = (0.24 - 2.45) / (0.1 - 0.0)
             |   \                      = -0.21 / 0.1
             |    \                     = -2.1
             |     \
        2.24 |      \
             -----------► m2
             0.0     0.1

We could do this for all 6 parameters and use the estimated slope to guide the gradient
descent learning process. However in practice this is expensive: we have to recompute
the output for each new parameter value we try. This also could be inaccurate since we
picked a fixed step size for how much we change the model parameters.

The idea of back-propagation is to apply the rules of calculus to compute equations for
our slopes in an efficient way. In calculus, our slope `𝚫L / 𝚫m2` is the partial
derivative `δL / δm2`, the rate of change of L with respect to m2.

The cross entropy loss equation is:

                                            e^(m2x + b2)
        Loss = -ln(ŷ2) = -ln(------------------------------------------)
                             e^(m1x + b1) + e^(m2x + b2) + e^(m3x + b3)

Taking the derivative of that is complex. Instead we can use the chain rule in calculus,
which is this: consider a simple example where we have 2 compute blocks, first `y=2x`
and second `z=4y`. The full system is `z=8x`, and the slope or derivative of this system
equation is 8.

The chain rule says that you can compute the same answer by computing the derivative of
each block individually. The slope of the first block is `dy/dx = 2`, and the slope of
the second is `dz/dy = 4`. We can multiply those rate of change together to get the
overall rate of change: `dz/dx = 8`.

Let's apply this to our problem. We break `δL / δm` into `δh / δm` (our linear model)
and `δL / δh` (the softmax function). It turns out that the logarithm from cross-entropy
loss and the exponentials from the softmax function basically cancel each other. The
result is:

         δL / δh = ŷ - y

where `ŷ` is a vector containing the 3 output probabilities, and `y` is a vector that
includes 1 at the index for Paris and 0 for everything else.

Let's apply this to our example:

| City   | h     | ŷ    | y | δL / δh = ŷ - y |
|--------|-------|------|---|-----------------|
| Madrid | 2.35  | 0.91 | 0 | 0.91            |
| Paris  | 0.00  | 0.09 | 1 | -0.91           |
| Berlin | -2.35 | 0.00 | 0 | 0.00            |

        δL / δm2 = (δh / δm2) . (δL / δh2)
                 =     x      . (ŷ2 - y2)
                 = 2.3514 . (0.09 - 1)
                 = 2.3514 . (-0.91)
                 = -2.14

Computed partial derivative values like this make up the gradient vector, which drive
the learning process:

        gradient = [δL/δm1, δL/δm2, δL/δm3, δL/δb1, δL/δb2, δL/δb3]
                 = [2.14  , -2.14,  0,      0.91,   -0.91,  0     ]

To train our model, we should update the parameters in the *opposite direction* to the
gradients. Specifically:

- m1 = m1 - α . (δL/δm1)
- m2 = m2 - α . (δL/δm2)
- etc.

Here `α` is he *learning rate* (typically small e.g. 0.0001). This is the **gradient
decent process**.

Now if we want to expend our model to also process latitude, the model takes 2 inputs x1
and x2 (latitude and longitude), and each neuron has now 3 parameters: m~i~1, m~i~2,
b~i~. For example, the equation for the neuron for Madrid is now:

        h1 = (m11 . x1) + (m12 . x2) + b1

Each neuron effectively learns a plane, not just a linear equation like before.

The model works like a very reduced version of Llama: each token in Llama is mapped to
an embedding vector of size 2048. GPS coordinates are an embedding vector of size 2.

## Deep learning

## ALEXNET

## Neural scaling laws

## Mechanistic interpretability

## Attention

## Diffusion
