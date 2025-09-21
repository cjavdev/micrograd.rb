# frozen_string_literal: true

require 'spec_helper'
require_relative '../lib/layer'

RSpec.describe Layer do
  describe '#initialize' do
    it 'creates a layer with the specified number of inputs and outputs' do
      layer = Layer.new(3, 2)
      expect(layer.instance_variable_get(:@neurons).length).to eq(2)
    end

    it 'creates neurons with the correct number of inputs' do
      layer = Layer.new(5, 3)
      neurons = layer.instance_variable_get(:@neurons)

      neurons.each do |neuron|
        expect(neuron.weights.length).to eq(5)
      end
    end

    it 'creates different neurons with random weights' do
      layer = Layer.new(2, 3)
      neurons = layer.instance_variable_get(:@neurons)

      # Each neuron should have different weights (extremely unlikely to be identical)
      weight_vectors = neurons.map { |neuron| neuron.weights.map(&:data) }

      # Check that at least one weight vector is different from others
      all_same = weight_vectors.all? { |weights| weights == weight_vectors[0] }
      expect(all_same).to be false
    end

    it 'handles single output layer' do
      layer = Layer.new(4, 1)
      expect(layer.instance_variable_get(:@neurons).length).to eq(1)
    end

    it 'handles single input layer' do
      layer = Layer.new(1, 3)
      neurons = layer.instance_variable_get(:@neurons)

      neurons.each do |neuron|
        expect(neuron.weights.length).to eq(1)
      end
    end
  end

  describe '#call' do
    let(:layer) { Layer.new(3, 2) }

    it 'processes input and returns an array of Value objects' do
      input = [1.0, 2.0, 3.0]
      output = layer.call(input)

      expect(output).to be_an(Array)
      expect(output.length).to eq(2)
      output.each do |value|
        expect(value).to be_a(Value)
      end
    end

    it 'returns outputs from all neurons' do
      input = [1.0, 2.0, 3.0]
      output = layer.call(input)

      expect(output.length).to eq(2)

      # Each output should be a tanh activation (between -1 and 1)
      output.each do |value|
        expect(value.data).to be_between(-1.0, 1.0)
      end
    end

    it 'handles numeric inputs by converting them to Value objects' do
      input = [1.0, 2.0, 3.0]
      output = layer.call(input)

      output.each do |value|
        expect(value.data).to be_a(Float)
      end
    end

    it 'handles Value object inputs directly' do
      input = [Value.new(1.0), Value.new(2.0), Value.new(3.0)]
      output = layer.call(input)

      output.each do |value|
        expect(value).to be_a(Value)
      end
    end

    it 'handles mixed input types' do
      input = [1.0, Value.new(2.0), 3.0]
      output = layer.call(input)

      output.each do |value|
        expect(value).to be_a(Value)
      end
    end

    it 'processes each neuron independently' do
      # Create a layer with known neurons for testing
      layer = Layer.new(2, 2)

      # Set specific weights and biases for predictable testing
      neurons = layer.instance_variable_get(:@neurons)
      neurons[0].instance_variable_set(:@w, [Value.new(0.5), Value.new(-0.3)])
      neurons[0].instance_variable_set(:@b, Value.new(0.1))
      neurons[1].instance_variable_set(:@w, [Value.new(-0.2), Value.new(0.8)])
      neurons[1].instance_variable_set(:@b, Value.new(-0.4))

      input = [1.0, 2.0]
      output = layer.call(input)

      # Neuron 0: (0.5 * 1.0) + (-0.3 * 2.0) + 0.1 = 0.0, tanh(0.0) = 0.0
      # Neuron 1: (-0.2 * 1.0) + (0.8 * 2.0) + (-0.4) = 1.0, tanh(1.0) ≈ 0.762
      expect(output[0].data).to be_within(0.0001).of(0.0)
      expect(output[1].data).to be_within(0.001).of(0.762)
    end

    it 'handles zero inputs' do
      input = [0.0, 0.0, 0.0]
      output = layer.call(input)

      output.each do |value|
        expect(value).to be_a(Value)
        expect(value.data).to be_between(-1.0, 1.0)
      end
    end

    it 'handles negative inputs' do
      input = [-1.0, -2.0, -3.0]
      output = layer.call(input)

      output.each do |value|
        expect(value).to be_a(Value)
        expect(value.data).to be_between(-1.0, 1.0)
      end
    end

    it 'handles large inputs' do
      input = [100.0, 200.0, 300.0]
      output = layer.call(input)

      output.each do |value|
        expect(value).to be_a(Value)
        expect(value.data).to be_between(-1.0, 1.0)
      end
    end

    it 'maintains computation graph for gradient computation' do
      input = [Value.new(1.0), Value.new(2.0), Value.new(3.0)]
      output = layer.call(input)

      output.each do |value|
        expect(value).to respond_to(:grad)
        expect(value).to respond_to(:left)
        expect(value).to respond_to(:right)
        expect(value).to respond_to(:op)
        expect(value.op).to eq(:tanh)
      end
    end
  end

  describe 'gradient computation' do
    it 'allows backpropagation through the layer' do
      layer = Layer.new(2, 2)
      input = [Value.new(1.0), Value.new(2.0)]
      output = layer.call(input)

      # Initialize gradients
      output.each { |value| value.grad = 1.0 }

      # Backward pass should work without errors
      output.each do |value|
        expect { value.backward }.not_to raise_error
      end
    end

    it 'computes gradients for all neurons independently' do
      layer = Layer.new(2, 2)
      input = [Value.new(1.0), Value.new(2.0)]
      output = layer.call(input)

      # Set different gradients for each output
      output[0].grad = 1.0
      output[1].grad = 2.0

      # Backward pass
      output.each { |value| value.backward }

      # Check that gradients were computed
      neurons = layer.instance_variable_get(:@neurons)
      neurons.each do |neuron|
        expect(neuron.weights.map(&:grad)).to all(be_a(Float))
        expect(neuron.bias.grad).to be_a(Float)
      end
    end

    it 'accumulates gradients correctly when multiple outputs share inputs' do
      layer = Layer.new(1, 2)
      input = [Value.new(2.0)]
      output = layer.call(input)

      # Both outputs use the same input, so gradients should accumulate
      output[0].grad = 1.0
      output[1].grad = 1.0

      output.each { |value| value.backward }

      # Input gradient should be accumulated from both outputs (could be positive or negative)
      expect(input[0].grad).not_to eq(0.0)
    end
  end

  describe 'edge cases' do
    it 'handles single input single output layer' do
      layer = Layer.new(1, 1)
      input = [1.0]
      output = layer.call(input)

      expect(output).to be_a(Value)
    end

    it 'handles many inputs and outputs' do
      layer = Layer.new(10, 5)
      input = Array.new(10, 1.0)
      output = layer.call(input)

      expect(output.length).to eq(5)
      output.each do |value|
        expect(value).to be_a(Value)
        expect(value.data).to be_between(-1.0, 1.0)
      end
    end

    it 'handles empty input array gracefully' do
      layer = Layer.new(0, 2)
      input = []
      output = layer.call(input)

      expect(output.length).to eq(2)
      output.each do |value|
        expect(value).to be_a(Value)
        expect(value.data).to be_between(-1.0, 1.0)
      end
    end
  end

  describe 'mathematical properties' do
    it 'preserves tanh bounds for all outputs' do
      layer = Layer.new(3, 4)

      test_inputs = [
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [-1.0, -1.0, -1.0],
        [10.0, 10.0, 10.0],
        [-10.0, -10.0, -10.0]
      ]

      test_inputs.each do |input|
        output = layer.call(input)
        output.each do |value|
          expect(value.data).to be_between(-1.0, 1.0)
        end
      end
    end

    it 'is deterministic with same inputs and layer configuration' do
      layer = Layer.new(2, 2)

      # Set specific weights and biases
      neurons = layer.instance_variable_get(:@neurons)
      neurons[0].instance_variable_set(:@w, [Value.new(0.5), Value.new(-0.3)])
      neurons[0].instance_variable_set(:@b, Value.new(0.1))
      neurons[1].instance_variable_set(:@w, [Value.new(-0.2), Value.new(0.8)])
      neurons[1].instance_variable_set(:@b, Value.new(-0.4))

      input = [1.0, 2.0]
      output1 = layer.call(input)
      output2 = layer.call(input)

      expect(output1.length).to eq(output2.length)
      output1.zip(output2).each do |val1, val2|
        expect(val1.data).to eq(val2.data)
      end
    end
  end

  describe 'integration with Value class' do
    it 'works with Value arithmetic operations' do
      layer = Layer.new(2, 2)
      input = [Value.new(1.0), Value.new(2.0)]
      output = layer.call(input)

      # Should be able to perform operations with the outputs
      combined = output[0] + output[1]
      expect(combined).to be_a(Value)

      squared = output[0] * output[0]
      expect(squared).to be_a(Value)
    end

    it 'maintains computation graph for backpropagation' do
      layer = Layer.new(2, 2)
      input = [Value.new(1.0), Value.new(2.0)]
      output = layer.call(input)

      # The computation graph should be preserved
      output.each do |value|
        expect(value).to respond_to(:left)
        expect(value).to respond_to(:op)
        expect(value.op).to eq(:tanh)
      end
    end

    it 'allows chaining with other layers or operations' do
      layer1 = Layer.new(2, 3)
      layer2 = Layer.new(3, 1)

      input = [Value.new(1.0), Value.new(2.0)]
      hidden = layer1.call(input)
      output = layer2.call(hidden)

      expect(output).to be_a(Value)

      # Should support backpropagation through the chain
      output.grad = 1.0
      expect { output.backward }.not_to raise_error
    end
  end

  describe 'neuron access' do
    it 'allows access to individual neurons' do
      layer = Layer.new(2, 3)
      neurons = layer.instance_variable_get(:@neurons)

      expect(neurons.length).to eq(3)
      neurons.each do |neuron|
        expect(neuron).to be_a(Neuron)
        expect(neuron.weights.length).to eq(2)
      end
    end

    it 'allows parameter access through neurons' do
      layer = Layer.new(2, 2)
      neurons = layer.instance_variable_get(:@neurons)

      all_params = neurons.flat_map(&:parameters)
      expect(all_params.length).to eq(6) # 2 neurons * (2 weights + 1 bias)

      all_params.each do |param|
        expect(param).to be_a(Value)
      end
    end
  end

  describe 'performance characteristics' do
    it 'handles reasonable layer sizes efficiently' do
      layer = Layer.new(100, 50)
      input = Array.new(100, 1.0)

      start_time = Time.now
      output = layer.call(input)
      end_time = Time.now

      expect(output.length).to eq(50)
      expect(end_time - start_time).to be < 1.0 # Should complete in under 1 second
    end
  end
end
