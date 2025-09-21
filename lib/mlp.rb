require_relative 'layer'
require 'byebug'

class MLP
  def initialize(number_of_inputs, sizes_of_layers)
    # Build layers with correct input/output dimensions
    sizes = [number_of_inputs] + sizes_of_layers
    @layers = []

    (0...sizes_of_layers.length).each do |i|
      @layers << Layer.new(sizes[i], sizes[i + 1])
    end
  end

  def call(x)
    # Forward pass through all layers
    @layers.reduce(x) { |x, layer| layer.call(x) }
  end

  def parameters
    @layers.map(&:parameters).flatten
  end

  def layers
    @layers
  end
end