require_relative 'neuron'

class Layer
  def initialize(number_of_inputs, number_of_outputs)
    @neurons = number_of_outputs.times.map { Neuron.new(number_of_inputs) }
  end

  def call(x)
    r = @neurons.map { |neuron| neuron.call(x) }
    if r.length == 1
      r[0]
    else
      r
    end
  end

  def parameters
    @neurons.map { |neuron| neuron.parameters }.flatten
  end
end