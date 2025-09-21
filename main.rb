require_relative 'lib/value'
require_relative 'lib/neuron'
require_relative 'lib/layer'
require_relative 'lib/mlp'
require_relative 'lib/render'


# Test out the value Class:
#
# x1 = Value.new(2.0, label: 'x1')
# x2 = Value.new(0.0, label: 'x2')

# w1 = Value.new(-3.0, label: 'w1')
# w2 = Value.new(1.0, label: 'w2')
# b = Value.new(6.881373587019543, label: 'b')


# x1w1 = x1 * w1
# x1w1.label = 'x1w1'
# x2w2 = x2 * w2
# x2w2.label = 'x2w2'
# x1w1x2w2 = x1w1 + x2w2
# x1w1x2w2.label = 'x1w1x2w2'
# n = x1w1x2w2 + b
# n.label = 'n'

# o = n.tanh()
# o.label = 'o'

# o.grad = 1.0
# o.backward
# Render.new(o).render


# Test out the neuron Class:
# srand(1)
# n = Neuron.new(2)
# result = n.call([1.0, 2.0])
# result.backward
# Render.new(result).render

# Test out the layer Class:
#
# l = Layer.new(2, 1)
# result = l.call([1.0, 2.0])
# p result
# Render.new(result).render

# Test out the mlp Class:

mlp = MLP.new(3, [4, 4, 1])

xs = [
  [2.0, 3.0, -1],
  [3, -1, 0.5],
  [0.5, 1.0, 1.0],
  [1.0, 1.0, -1.0]
]
ys = [-1.0, -1.0, 1.0, -1.0]

loss = nil
# Training loop
70.times do |k|
  # Forward pass.
  ypred = xs.map { |x| mlp.call(x) }
  # p ypred.map(&:data)
  loss = ypred
    .zip(ys)
    .map { |y_predicted, y_ground_truth| (y_predicted - y_ground_truth)**2 }
    .sum

  # Backward pass
  mlp.parameters.each { |p| p.grad = 0.0 }
  loss.backward

  # # Update weights
  mlp.parameters.each { |p| p.data += -0.05 * p.grad }

  # Print loss
  puts "Step #{k} loss: #{loss.data}"
end

Render.new(loss).render

ypred = xs.map { |x| mlp.call(x) }
p ypred.map(&:data)