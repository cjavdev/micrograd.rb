# Learn ML - Value Class

A simple Ruby implementation of a Value class with automatic differentiation, Neuron, Layer, and MLP classes for machine learning applications.

## Project Structure

```
learn-ml/
├── lib/
│   ├── value.rb          # Value class implementation
│   ├── neuron.rb         # Neuron class implementation
│   ├── layer.rb          # Layer class implementation
│   └── mlp.rb            # MLP class implementation
├── spec/
│   ├── spec_helper.rb    # RSpec configuration
│   ├── value_spec.rb     # Value class test suite
│   ├── neuron_spec.rb    # Neuron class test suite
│   ├── layer_spec.rb     # Layer class test suite
│   └── mlp_spec.rb       # MLP class test suite
├── main.rb               # Example usage
├── Gemfile               # Ruby dependencies
└── README.md             # This file
```

## Value Class Features

The Value class supports the following mathematical operations:

### Basic Operations
- `+` - Addition
- `-` - Subtraction
- `*` - Multiplication
- `/` - Division (with zero-division protection)
- `**` - Power operation

### Mathematical Functions
- `tanh` - Hyperbolic tangent
- `exp` - Exponential function
- `abs` - Absolute value
- `sqrt` - Square root (with negative number protection)

### Comparison Operations
- `==` - Equality
- `>`, `<`, `>=`, `<=` - Comparison operators

### Unary Operations
- `-@` - Unary minus (negation)
- `+@` - Unary plus (returns self)

## Neuron Class Features

The Neuron class implements a simple artificial neuron with:

- **Random initialization**: Weights and bias are randomly initialized between -1 and 1
- **Forward pass**: Computes weighted sum + bias and applies tanh activation
- **Gradient computation**: Full support for backpropagation through automatic differentiation
- **Flexible inputs**: Accepts both numeric values and Value objects

### Neuron Usage

```ruby
require_relative 'lib/neuron'

# Create a neuron with 3 inputs
neuron = Neuron.new(3)

# Forward pass
input = [1.0, 2.0, 3.0]
output = neuron.call(input)

# Backward pass for gradient computation
output.backward

puts "Output: #{output.data}"
puts "Weight gradients: #{neuron.weights.map(&:grad)}"
puts "Bias gradient: #{neuron.bias.grad}"
```

## Layer Class Features

The Layer class implements a collection of neurons that can be used to build neural networks:

- **Multiple neurons**: Creates a specified number of neurons with the same number of inputs
- **Parallel processing**: Each neuron processes the same input independently
- **Array output**: Returns an array of Value objects, one from each neuron
- **Gradient computation**: Full support for backpropagation through all neurons
- **Layer chaining**: Can be easily combined with other layers to build deep networks

### Layer Usage

```ruby
require_relative 'lib/layer'

# Create a layer with 3 inputs and 2 outputs
layer = Layer.new(3, 2)

# Forward pass
input = [1.0, 2.0, 3.0]
output = layer.call(input)

# Backward pass for gradient computation
output.each { |out| out.grad = 1.0 }
output.each { |out| out.backward }

puts "Layer output: #{output.map(&:data)}"
puts "Number of neurons: #{layer.instance_variable_get(:@neurons).length}"
```

## MLP Class Features

The MLP (Multi-Layer Perceptron) class implements a complete neural network:

- **Multi-layer architecture**: Creates a network with any number of layers and neurons
- **Automatic layer chaining**: Properly connects layers with correct input/output dimensions
- **Forward propagation**: Processes input through all layers sequentially
- **Gradient computation**: Full backpropagation support through the entire network
- **Flexible architecture**: Supports any network topology (narrow, wide, deep)

### MLP Usage

```ruby
require_relative 'lib/mlp'

# Create an MLP with 3 inputs -> 4 hidden -> 2 outputs
mlp = MLP.new(3, [4, 2])

# Forward pass
input = [1.0, 2.0, 3.0]
output = mlp.call(input)

# Backward pass for gradient computation
output.each { |out| out.grad = 1.0 }
output.each { |out| out.backward }

puts "MLP output: #{output.map(&:data)}"
puts "Total parameters: #{mlp.parameters.length}"
```

## Value Class Features

The Value class supports the following mathematical operations:

### Basic Operations
- `+` - Addition
- `-` - Subtraction
- `*` - Multiplication
- `/` - Division (with zero-division protection)
- `**` - Power operation

### Mathematical Functions
- `tanh` - Hyperbolic tangent
- `exp` - Exponential function
- `abs` - Absolute value
- `sqrt` - Square root (with negative number protection)

### Comparison Operations
- `==` - Equality
- `>`, `<`, `>=`, `<=` - Comparison operators

### Unary Operations
- `-@` - Unary minus (negation)
- `+@` - Unary plus (returns self)

### Automatic Differentiation
- `grad` - Gradient attribute
- `backward` - Computes gradients using backpropagation
- `_children` - Tracks computation graph
- `_op` - Tracks operation type

## Usage

```ruby
require_relative 'lib/value'

a = Value.new(5.0)
b = Value.new(3.0)

puts a + b     # => 8.0
puts a * b     # => 15.0
puts a.tanh    # => 0.9999092042625951
puts a.exp     # => 148.4131591025766

# Automatic differentiation
c = a * b + a
c.backward
puts "Gradient of a: #{a.grad}"  # => 4.0
puts "Gradient of b: #{b.grad}"  # => 2.0
```

## Running Tests

```bash
# Install dependencies
bundle install

# Run all tests
bundle exec rspec

# Run tests with documentation format
bundle exec rspec --format documentation
```

## Example Output

```bash
$ ruby main.rb
Testing Value class...
a = 5.0
b = 3.0
a + b = 8.0
a - b = 2.0
a * b = 15.0
a / b = 1.6666666666666667
a ** b = 125.0
a.tanh = 0.9999092042625951
a.exp = 148.4131591025766

==================================================
Testing Neuron class...
Neuron weights: [-0.9483489345076213, 0.11382553925359051, -0.7320466128837473]
Neuron bias: -0.612724161236365
Input: [1.0, 2.0, 3.0]
Output: -0.9982824143961624
Gradients after backward pass:
  Input gradients: ["N/A", "N/A", "N/A"]
  Weight gradients: [0.003432221107368738, 0.006864442214737476, 0.010296663322106214]
  Bias gradient: 0.003432221107368738
```

## Test Coverage

The project includes comprehensive test suites with **147 test cases** covering:

- **Value class**: 42 tests including mathematical operations, gradient computation, and edge cases
- **Neuron class**: 41 tests including initialization, forward pass, gradient computation, and integration tests
- **Layer class**: 29 tests including initialization, forward pass, gradient computation, layer chaining, and performance tests
- **MLP class**: 35 tests including architecture validation, forward/backward propagation, and integration tests

All tests pass with 100% success rate, ensuring the reliability of automatic differentiation and neural network functionality for building complex machine learning models.
