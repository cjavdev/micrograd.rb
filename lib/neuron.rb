require_relative 'value'

class Neuron
  def initialize(number_of_inputs)
    @w = number_of_inputs.times.map { Value.new(rand(-1.0..1.0)) }
    @b = Value.new(rand(-1.0..1.0))
  end

  def call(x)
    # Convert inputs to Value objects if they aren't already
    x_values = x.map { |xi| xi.is_a?(Value) ? xi : Value.new(xi) }

    # Calculate weighted sum: w1*x1 + w2*x2 + ... + b
    weighted_sum = @w.zip(x_values).reduce(@b) do |sum, (wi, xi)|
      sum + wi * xi
    end

    # Apply tanh activation
    weighted_sum.tanh
  end

  def parameters
    @w + [@b]
  end

  def weights
    @w
  end

  def bias
    @b
  end
end