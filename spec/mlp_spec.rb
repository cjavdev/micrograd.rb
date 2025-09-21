# frozen_string_literal: true

require 'spec_helper'
require_relative '../lib/mlp'

RSpec.describe MLP do
  describe '#initialize' do
    it 'creates an MLP with the specified architecture' do
      mlp = MLP.new(3, [4, 2])
      expect(mlp.layers.length).to eq(2)
    end

    it 'creates layers with correct input/output dimensions' do
      mlp = MLP.new(3, [4, 2])

      # First layer: 3 inputs -> 4 outputs
      expect(mlp.layers[0].instance_variable_get(:@neurons).length).to eq(4)
      mlp.layers[0].instance_variable_get(:@neurons).each do |neuron|
        expect(neuron.weights.length).to eq(3)
      end

      # Second layer: 4 inputs -> 2 outputs
      expect(mlp.layers[1].instance_variable_get(:@neurons).length).to eq(2)
      mlp.layers[1].instance_variable_get(:@neurons).each do |neuron|
        expect(neuron.weights.length).to eq(4)
      end
    end

    it 'handles single layer MLP' do
      mlp = MLP.new(2, [3])
      expect(mlp.layers.length).to eq(1)
      expect(mlp.layers[0].instance_variable_get(:@neurons).length).to eq(3)
      mlp.layers[0].instance_variable_get(:@neurons).each do |neuron|
        expect(neuron.weights.length).to eq(2)
      end
    end

    it 'handles deep MLP with many layers' do
      mlp = MLP.new(2, [4, 8, 4, 1])
      expect(mlp.layers.length).to eq(4)

      expected_sizes = [[2, 4], [4, 8], [8, 4], [4, 1]]
      expected_sizes.each_with_index do |(input_size, output_size), i|
        expect(mlp.layers[i].instance_variable_get(:@neurons).length).to eq(output_size)
        mlp.layers[i].instance_variable_get(:@neurons).each do |neuron|
          expect(neuron.weights.length).to eq(input_size)
        end
      end
    end

    it 'creates different random weights for each layer' do
      mlp = MLP.new(2, [3, 2])

      # Each layer should have different random weights
      layer1_weights = mlp.layers[0].instance_variable_get(:@neurons).map(&:weights)
      layer2_weights = mlp.layers[1].instance_variable_get(:@neurons).map(&:weights)

      # It's extremely unlikely that all weights would be identical
      all_same = layer1_weights.flatten.zip(layer2_weights.flatten).all? do |w1, w2|
        (w1.data - w2.data).abs < 0.0001
      end
      expect(all_same).to be false
    end
  end

  describe '#call' do
    let(:mlp) { MLP.new(3, [4, 2]) }

    it 'processes input through all layers and returns final output' do
      input = [1.0, 2.0, 3.0]
      output = mlp.call(input)

      expect(output).to be_an(Array)
      expect(output.length).to eq(2) # Final layer output size
      output.each do |value|
        expect(value).to be_a(Value)
      end
    end

    it 'applies tanh activation at each layer' do
      input = [1.0, 2.0, 3.0]
      output = mlp.call(input)

      # All outputs should be between -1 and 1 (tanh bounds)
      output.each do |value|
        expect(value.data).to be_between(-1.0, 1.0)
      end
    end

    it 'handles numeric inputs by converting them to Value objects' do
      input = [1.0, 2.0, 3.0]
      output = mlp.call(input)

      output.each do |value|
        expect(value.data).to be_a(Float)
      end
    end

    it 'handles Value object inputs directly' do
      input = [Value.new(1.0), Value.new(2.0), Value.new(3.0)]
      output = mlp.call(input)

      output.each do |value|
        expect(value).to be_a(Value)
      end
    end

    it 'handles mixed input types' do
      input = [1.0, Value.new(2.0), 3.0]
      output = mlp.call(input)

      output.each do |value|
        expect(value).to be_a(Value)
      end
    end

    it 'processes through multiple layers correctly' do
      # Create a simple 2-layer MLP with known weights for testing
      mlp = MLP.new(2, [2, 1])

      # Set specific weights for predictable testing
      layer1 = mlp.layers[0]
      layer1.instance_variable_get(:@neurons)[0].instance_variable_set(:@w, [Value.new(0.5), Value.new(-0.3)])
      layer1.instance_variable_get(:@neurons)[0].instance_variable_set(:@b, Value.new(0.1))
      layer1.instance_variable_get(:@neurons)[1].instance_variable_set(:@w, [Value.new(-0.2), Value.new(0.8)])
      layer1.instance_variable_get(:@neurons)[1].instance_variable_set(:@b, Value.new(-0.4))

      layer2 = mlp.layers[1]
      layer2.instance_variable_get(:@neurons)[0].instance_variable_set(:@w, [Value.new(0.6), Value.new(-0.7)])
      layer2.instance_variable_get(:@neurons)[0].instance_variable_set(:@b, Value.new(0.2))

      input = [1.0, 2.0]
      output = mlp.call(input)

      expect(output).to be_a(Value)
      expect(output.data).to be_between(-1.0, 1.0)
    end

    it 'handles zero inputs' do
      input = [0.0, 0.0, 0.0]
      output = mlp.call(input)

      output.each do |value|
        expect(value).to be_a(Value)
        expect(value.data).to be_between(-1.0, 1.0)
      end
    end

    it 'handles negative inputs' do
      input = [-1.0, -2.0, -3.0]
      output = mlp.call(input)

      output.each do |value|
        expect(value).to be_a(Value)
        expect(value.data).to be_between(-1.0, 1.0)
      end
    end

    it 'handles large inputs' do
      input = [100.0, 200.0, 300.0]
      output = mlp.call(input)

      output.each do |value|
        expect(value).to be_a(Value)
        expect(value.data).to be_between(-1.0, 1.0)
      end
    end

    it 'maintains computation graph for gradient computation' do
      input = [Value.new(1.0), Value.new(2.0), Value.new(3.0)]
      output = mlp.call(input)

      output.each do |value|
        expect(value).to respond_to(:grad)
        expect(value).to respond_to(:left)
        expect(value).to respond_to(:right)
        expect(value).to respond_to(:op)
        expect(value.op).to eq(:tanh)
      end
    end
  end

  describe '#parameters' do
    it 'returns all parameters from all layers' do
      mlp = MLP.new(3, [4, 2])
      params = mlp.parameters

      # Should have: (3*4 + 4) + (4*2 + 2) = 16 + 10 = 26 parameters
      expected_params = (3 * 4 + 4) + (4 * 2 + 2)
      expect(params.length).to eq(expected_params)

      params.each do |param|
        expect(param).to be_a(Value)
      end
    end

    it 'returns parameters from multiple layers' do
      mlp = MLP.new(2, [3, 4, 1])
      params = mlp.parameters

      # Should have: (2*3 + 3) + (3*4 + 4) + (4*1 + 1) = 9 + 16 + 5 = 30 parameters
      expected_params = (2 * 3 + 3) + (3 * 4 + 4) + (4 * 1 + 1)
      expect(params.length).to eq(expected_params)
    end

    it 'returns parameters from single layer' do
      mlp = MLP.new(3, [2])
      params = mlp.parameters

      # Should have: (3*2 + 2) = 8 parameters
      expected_params = 3 * 2 + 2
      expect(params.length).to eq(expected_params)
    end
  end

  describe '#layers' do
    it 'returns the layers array' do
      mlp = MLP.new(3, [4, 2])
      layers = mlp.layers

      expect(layers).to be_an(Array)
      expect(layers.length).to eq(2)
      layers.each do |layer|
        expect(layer).to be_a(Layer)
      end
    end
  end

  describe 'gradient computation' do
    it 'allows backpropagation through the entire MLP' do
      mlp = MLP.new(2, [3, 1])
      input = [Value.new(1.0), Value.new(2.0)]
      output = mlp.call(input)

      # Initialize gradients
      output.grad = 1.0

      # Backward pass should work without errors
      expect { output.backward }.not_to raise_error
    end

    it 'computes gradients for all layers' do
      mlp = MLP.new(2, [2, 1])
      input = [Value.new(1.0), Value.new(2.0)]
      output = mlp.call(input)

      # Set gradient
      output.grad = 1.0

      # Backward pass
      output.backward

      # Check that gradients were computed for all layers
      mlp.layers.each do |layer|
        layer.instance_variable_get(:@neurons).each do |neuron|
          expect(neuron.weights.map(&:grad)).to all(be_a(Float))
          expect(neuron.bias.grad).to be_a(Float)
        end
      end
    end

    it 'accumulates gradients correctly through multiple layers' do
      mlp = MLP.new(1, [2, 1])
      input = [Value.new(2.0)]
      output = mlp.call(input)

      # Set gradient
      output.grad = 1.0

      # Backward pass
      output.backward

      # Input gradient should be computed and non-zero
      expect(input[0].grad).not_to eq(0.0)
    end

    it 'handles multiple output gradients' do
      mlp = MLP.new(2, [3, 2])
      input = [Value.new(1.0), Value.new(2.0)]
      output = mlp.call(input)

      # Set different gradients for each output
      output[0].grad = 1.0
      output[1].grad = 2.0

      # Backward pass
      output.each { |out| out.backward }

      # All parameters should have gradients
      mlp.parameters.each do |param|
        expect(param.grad).to be_a(Float)
      end
    end
  end

  describe 'edge cases' do
    it 'handles single input single output MLP' do
      mlp = MLP.new(1, [1])
      input = [1.0]
      output = mlp.call(input)

      expect(output).to be_a(Value)
    end

    it 'handles many inputs and outputs' do
      mlp = MLP.new(10, [5, 3])
      input = Array.new(10, 1.0)
      output = mlp.call(input)

      expect(output.length).to eq(3)
      output.each do |value|
        expect(value).to be_a(Value)
        expect(value.data).to be_between(-1.0, 1.0)
      end
    end

    it 'handles empty layer sizes array' do
      mlp = MLP.new(3, [])
      input = [1.0, 2.0, 3.0]
      output = mlp.call(input)

      # Should return the input unchanged (no layers)
      expect(output).to eq(input)
    end
  end

  describe 'mathematical properties' do
    it 'preserves tanh bounds for all outputs' do
      mlp = MLP.new(3, [4, 2])

      test_inputs = [
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [-1.0, -1.0, -1.0],
        [10.0, 10.0, 10.0],
        [-10.0, -10.0, -10.0]
      ]

      test_inputs.each do |input|
        output = mlp.call(input)
        output.each do |value|
          expect(value.data).to be_between(-1.0, 1.0)
        end
      end
    end

    it 'is deterministic with same inputs and MLP configuration' do
      mlp = MLP.new(2, [2, 1])

      # Set specific weights for all layers
      mlp.layers.each do |layer|
        layer.instance_variable_get(:@neurons).each do |neuron|
          neuron.instance_variable_set(:@w, [Value.new(0.1), Value.new(0.2)])
          neuron.instance_variable_set(:@b, Value.new(0.3))
        end
      end

      input = [1.0, 2.0]
      output1 = mlp.call(input)
      output2 = mlp.call(input)

      expect(output1).to be_a(Value)
      expect(output2).to be_a(Value)
      output1.zip(output2).each do |val1, val2|
        expect(val1.data).to eq(val2.data)
      end
    end
  end

  describe 'integration with other classes' do
    it 'works with Value arithmetic operations' do
      mlp = MLP.new(2, [3, 1])
      input = [Value.new(1.0), Value.new(2.0)]
      output = mlp.call(input)

      # Should be able to perform operations with the output
      doubled = output[0] + output[0]
      expect(doubled).to be_a(Value)

      squared = output[0] * output[0]
      expect(squared).to be_a(Value)
    end

    it 'maintains computation graph for backpropagation' do
      mlp = MLP.new(2, [2, 1])
      input = [Value.new(1.0), Value.new(2.0)]
      output = mlp.call(input)

      # The computation graph should be preserved
      output.each do |value|
        expect(value).to respond_to(:left)
        expect(value).to respond_to(:right)
        expect(value).to respond_to(:op)
        expect(value.op).to eq(:tanh)
      end
    end

    it 'can be used in loss functions' do
      mlp = MLP.new(2, [3, 1])
      input = [Value.new(1.0), Value.new(2.0)]
      target = Value.new(0.5)

      output = mlp.call(input)
      loss = (output[0] - target) ** Value.new(2)

      expect(loss).to be_a(Value)
      expect(loss.data).to be >= 0

      # Should support backpropagation through loss
      loss.grad = 1.0
      expect { loss.backward }.not_to raise_error
    end
  end

  describe 'architecture variations' do
    it 'handles narrow MLPs' do
      mlp = MLP.new(5, [3, 2, 1])
      input = Array.new(5, 1.0)
      output = mlp.call(input)

      expect(output.length).to eq(1)
      expect(output[0]).to be_a(Value)
    end

    it 'handles wide MLPs' do
      mlp = MLP.new(2, [10, 20, 5])
      input = [1.0, 2.0]
      output = mlp.call(input)

      expect(output.length).to eq(5)
      output.each do |value|
        expect(value).to be_a(Value)
      end
    end

    it 'handles MLPs with same input/output dimensions' do
      mlp = MLP.new(3, [3, 3])
      input = [1.0, 2.0, 3.0]
      output = mlp.call(input)

      expect(output.length).to eq(3)
      output.each do |value|
        expect(value).to be_a(Value)
      end
    end
  end

  describe 'performance characteristics' do
    it 'handles reasonable MLP sizes efficiently' do
      mlp = MLP.new(10, [20, 10, 5])
      input = Array.new(10, 1.0)

      start_time = Time.now
      output = mlp.call(input)
      end_time = Time.now

      expect(output.length).to eq(5)
      expect(end_time - start_time).to be < 2.0 # Should complete in under 2 seconds
    end
  end
end
