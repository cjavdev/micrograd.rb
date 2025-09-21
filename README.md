# Learn ML - Neural Network Library

Based on https://www.youtube.com/watch?v=VMj-3S1tku0

A Ruby implementation of automatic differentiation and neural network building blocks using lambda-based backpropagation, similar to PyTorch's approach.

## Project Structure

```
learn-ml/
├── lib/
│   ├── value.rb          # Value class with lambda-based autograd
│   ├── neuron.rb         # Neuron class implementation
│   ├── layer.rb          # Layer class implementation
│   ├── mlp.rb            # MLP (Multi-Layer Perceptron) class
│   └── render.rb         # Graph visualization using GraphViz
├── spec/
│   ├── spec_helper.rb    # RSpec configuration
│   ├── value_spec.rb     # Value class test suite (42 tests)
│   ├── neuron_spec.rb    # Neuron class test suite (41 tests)
│   ├── layer_spec.rb     # Layer class test suite (29 tests)
│   └── mlp_spec.rb       # MLP class test suite (35 tests)
├── main.rb               # Example usage and demonstrations
├── Gemfile               # Ruby dependencies
├── graph.png             # Generated computation graph visualization
└── README.md             # This file
```

## Value Class - Lambda-Based Automatic Differentiation

The Value class implements automatic differentiation using **lambda-based backpropagation**. Each operation creates a result Value object and stores a lambda function that knows how to compute gradients for that specific operation.

### Key Features

- **Lambda-based gradients**: Each operation stores its own backward function as a lambda
- **Automatic graph traversal**: Uses topological sort for efficient gradient computation
- **Memory efficient**: Only stores necessary backward functions, not the entire computation graph
- **Type coercion**: Automatically converts numeric types to Value objects
- **Random labels**: Automatically generates 3-letter labels for visualization

### Mathematical Operations

```ruby
require_relative 'lib/value'

# Basic arithmetic operations
a = Value.new(2.0, label: 'a')
b = Value.new(-3.0, label: 'b')

# All operations return Value objects with stored backward functions
c = a + b        # Addition
d = a * b        # Multiplication
e = a / b        # Division
f = a ** 2       # Power (numeric exponent only)
g = a.tanh       # Hyperbolic tangent
h = a.exp        # Exponential

# Unary operations
neg_a = -a       # Negation (implemented as a * -1)
```

### Lambda-Based Backpropagation

```ruby
# Create a computation graph
a = Value.new(2.0, label: 'a')
b = Value.new(-3.0, label: 'b')
c = Value.new(10.0, label: 'c')

# Build expression: (a * b + c) * f
e = a * b; e.label = 'e'
d = e + c; d.label = 'd'
f = Value.new(-2.0, label: 'f')
L = d * f; L.label = 'L'

# Automatic differentiation using lambda functions
L.backward

puts "Gradient of b: #{b.grad}"  # => -4.0
puts "Gradient of f: #{f.grad}"  # => 4.0
```

### How Lambda-Based Autograd Works

1. **Forward Pass**: Each operation creates a result Value and stores a lambda function
2. **Lambda Storage**: The lambda is stored on the result Value (`out._backward = -> { ... }`)
3. **Variable Capture**: The lambda captures the operands in its closure
4. **Backward Pass**: `backward()` traverses the graph and calls each lambda
5. **Gradient Accumulation**: Each lambda updates gradients for its captured operands

```ruby
# Example: Addition operation
def +(other)
  other = other.is_a?(Value) ? other : Value.new(other)
  out = Value.new(@data + other.data, left: self, right: other, op: :+)

  # Store backward function on result, capturing operands
  out._backward = -> do
    self.grad += out.grad    # ∂(a+b)/∂a = 1
    other.grad += out.grad   # ∂(a+b)/∂b = 1
  end

  out
end
```

## Neuron Class

A simple artificial neuron with tanh activation and automatic gradient computation.

### Features

- **Random initialization**: Weights and bias initialized between -1 and 1
- **Tanh activation**: Applies hyperbolic tangent to weighted sum + bias
- **Flexible inputs**: Accepts numeric values or Value objects
- **Gradient support**: Full backpropagation through the neuron

### Usage

```ruby
require_relative 'lib/neuron'

# Create neuron with 3 inputs
neuron = Neuron.new(3)

# Forward pass
input = [1.0, 2.0, 3.0]
output = neuron.call(input)

# Backward pass
output.backward

puts "Output: #{output.data}"
puts "Weight gradients: #{neuron.weights.map(&:grad)}"
puts "Bias gradient: #{neuron.bias.grad}"
```

## Layer Class

A collection of neurons that process inputs in parallel.

### Features

- **Multiple neurons**: Creates specified number of neurons with same inputs
- **Parallel processing**: Each neuron processes input independently
- **Array output**: Returns array of Value objects
- **Layer chaining**: Easy composition with other layers

### Usage

```ruby
require_relative 'lib/layer'

# Create layer: 3 inputs → 2 outputs
layer = Layer.new(3, 2)

# Forward pass
input = [1.0, 2.0, 3.0]
output = layer.call(input)

# Backward pass
output.each { |out| out.grad = 1.0 }
output.each { |out| out.backward }

puts "Layer output: #{output.map(&:data)}"
```

## MLP Class (Multi-Layer Perceptron)

A complete neural network with multiple layers.

### Features

- **Flexible architecture**: Any number of layers and neurons per layer
- **Automatic chaining**: Properly connects layers with correct dimensions
- **Deep networks**: Supports arbitrary depth
- **Full backpropagation**: Gradients flow through entire network

### Usage

```ruby
require_relative 'lib/mlp'

# Create MLP: 3 inputs → 4 hidden → 2 outputs
mlp = MLP.new(3, [4, 2])

# Forward pass
input = [1.0, 2.0, 3.0]
output = mlp.call(input)

# Backward pass
output.each { |out| out.grad = 1.0 }
output.each { |out| out.backward }

puts "MLP output: #{output.map(&:data)}"
puts "Total parameters: #{mlp.parameters.length}"
```

## Graph Visualization

The library includes a Render class for visualizing computation graphs.

```ruby
require_relative 'lib/render'

# Create a computation graph
a = Value.new(2.0, label: 'a')
b = Value.new(-3.0, label: 'b')
c = a * b + Value.new(10.0, label: 'c')

# Generate graph visualization
render = Render.new(c)
render.render  # Creates graph.png
```

## Running Tests

```bash
# Install dependencies
bundle install

# Run all tests
bundle exec rspec

# Run with documentation format
bundle exec rspec --format documentation
```

## Test Coverage

**147 comprehensive test cases** covering:

- **Value class (42 tests)**: Mathematical operations, lambda-based gradients, edge cases
- **Neuron class (41 tests)**: Forward/backward pass, gradient computation, integration
- **Layer class (29 tests)**: Multi-neuron processing, gradient flow, chaining
- **MLP class (35 tests)**: Deep network architectures, end-to-end backpropagation

All tests pass with 100% success rate.

## Example Output

```bash
$ ruby main.rb
Value(-1.0, l: 2.0, r: -3.0, op: +, label: ABC)
```

## Key Advantages of Lambda-Based Approach

1. **Memory Efficient**: Only stores backward functions where needed
2. **Flexible**: Each operation defines its own gradient computation
3. **Modern**: Similar to PyTorch's autograd implementation
4. **Debuggable**: Easy to inspect and modify backward functions
5. **Extensible**: Simple to add new operations with custom gradients

## Dependencies

- **Ruby 3.4+**: For modern Ruby features
- **RSpec**: Testing framework
- **GraphViz**: For computation graph visualization

This implementation provides a solid foundation for building machine learning models in Ruby with automatic differentiation capabilities.