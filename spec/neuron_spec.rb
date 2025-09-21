# frozen_string_literal: true

require 'spec_helper'
require_relative '../lib/neuron'

RSpec.describe Neuron do
  describe '#initialize' do
    it 'creates a neuron with the specified number of inputs' do
      neuron = Neuron.new(3)
      expect(neuron.weights.length).to eq(3)
      expect(neuron.bias).to be_a(Value)
    end

    it 'initializes weights with random values between -1 and 1' do
      neuron = Neuron.new(2)
      neuron.weights.each do |weight|
        expect(weight.data).to be_between(-1.0, 1.0)
      end
    end

    it 'initializes bias with random value between -1 and 1' do
      neuron = Neuron.new(2)
      expect(neuron.bias.data).to be_between(-1.0, 1.0)
    end

    it 'creates different random weights for multiple instances' do
      neuron1 = Neuron.new(3)
      neuron2 = Neuron.new(3)

      # It's extremely unlikely that all weights would be identical
      weights_match = neuron1.weights.zip(neuron2.weights).all? do |w1, w2|
        (w1.data - w2.data).abs < 0.0001
      end
      expect(weights_match).to be false
    end
  end

  describe '#call' do
    let(:neuron) { Neuron.new(2) }

    it 'processes input and returns a Value object' do
      input = [1.0, 2.0]
      result = neuron.call(input)
      expect(result).to be_a(Value)
    end

    it 'applies tanh activation function' do
      input = [1.0, 2.0]
      result = neuron.call(input)
      # tanh output should be between -1 and 1
      expect(result.data).to be_between(-1.0, 1.0)
    end

    it 'handles numeric inputs by converting them to Value objects' do
      input = [1.0, 2.0]
      result = neuron.call(input)
      expect(result.data).to be_a(Float)
    end

    it 'handles Value object inputs directly' do
      input = [Value.new(1.0), Value.new(2.0)]
      result = neuron.call(input)
      expect(result).to be_a(Value)
    end

    it 'handles mixed input types' do
      input = [1.0, Value.new(2.0)]
      result = neuron.call(input)
      expect(result).to be_a(Value)
    end

    it 'calculates weighted sum correctly' do
      # Create a neuron with known weights and bias for testing
      neuron = Neuron.new(2)

      # Set specific weights and bias for predictable testing
      neuron.instance_variable_set(:@w, [Value.new(0.5), Value.new(-0.3)])
      neuron.instance_variable_set(:@b, Value.new(0.1))

      input = [1.0, 2.0]
      result = neuron.call(input)

      # Expected calculation: (0.5 * 1.0) + (-0.3 * 2.0) + 0.1 = 0.5 - 0.6 + 0.1 = 0.0
      # tanh(0.0) = 0.0
      expect(result.data).to be_within(0.0001).of(0.0)
    end

    it 'handles zero inputs' do
      input = [0.0, 0.0]
      result = neuron.call(input)
      # Result should be tanh(bias)
      expected = Math.tanh(neuron.bias.data)
      expect(result.data).to be_within(0.0001).of(expected)
    end

    it 'handles negative inputs' do
      input = [-1.0, -2.0]
      result = neuron.call(input)
      expect(result).to be_a(Value)
      expect(result.data).to be_between(-1.0, 1.0)
    end

    it 'handles large inputs' do
      input = [100.0, 200.0]
      result = neuron.call(input)
      expect(result).to be_a(Value)
      expect(result.data).to be_between(-1.0, 1.0)
    end
  end

  describe '#parameters' do
    it 'returns all weights and bias as an array' do
      neuron = Neuron.new(3)
      params = neuron.parameters

      expect(params.length).to eq(4) # 3 weights + 1 bias
      expect(params).to include(neuron.bias)
      neuron.weights.each do |weight|
        expect(params).to include(weight)
      end
    end

    it 'returns Value objects' do
      neuron = Neuron.new(2)
      params = neuron.parameters

      params.each do |param|
        expect(param).to be_a(Value)
      end
    end
  end

  describe '#weights' do
    it 'returns the weights array' do
      neuron = Neuron.new(3)
      weights = neuron.weights

      expect(weights).to be_an(Array)
      expect(weights.length).to eq(3)
      weights.each do |weight|
        expect(weight).to be_a(Value)
      end
    end
  end

  describe '#bias' do
    it 'returns the bias Value object' do
      neuron = Neuron.new(2)
      bias = neuron.bias

      expect(bias).to be_a(Value)
    end
  end

  describe 'gradient computation' do
    it 'maintains gradient information through forward pass' do
      neuron = Neuron.new(2)

      # Set specific weights and bias for testing
      neuron.instance_variable_set(:@w, [Value.new(0.5), Value.new(-0.3)])
      neuron.instance_variable_set(:@b, Value.new(0.1))

      input = [Value.new(1.0), Value.new(2.0)]
      result = neuron.call(input)

      # The result should have gradient information
      expect(result).to respond_to(:grad)
      expect(result).to respond_to(:left)
      expect(result).to respond_to(:right)
      expect(result).to respond_to(:op)
    end

    it 'allows backpropagation through the neuron' do
      neuron = Neuron.new(2)

      # Set specific weights and bias for testing
      neuron.instance_variable_set(:@w, [Value.new(0.5), Value.new(-0.3)])
      neuron.instance_variable_set(:@b, Value.new(0.1))

      input = [Value.new(1.0), Value.new(2.0)]
      result = neuron.call(input)

      # Initialize gradients
      result.grad = 1.0

      # Backward pass should work without errors
      expect { result.backward }.not_to raise_error
    end
  end

  describe 'edge cases' do
    it 'handles single input neuron' do
      neuron = Neuron.new(1)
      input = [1.0]
      result = neuron.call(input)

      expect(result).to be_a(Value)
      expect(result.data).to be_between(-1.0, 1.0)
    end

    it 'handles many inputs' do
      neuron = Neuron.new(10)
      input = Array.new(10, 1.0)
      result = neuron.call(input)

      expect(result).to be_a(Value)
      expect(result.data).to be_between(-1.0, 1.0)
    end

    it 'handles empty input array gracefully' do
      neuron = Neuron.new(0)
      input = []
      result = neuron.call(input)

      # Should just return tanh(bias)
      expected = Math.tanh(neuron.bias.data)
      expect(result.data).to be_within(0.0001).of(expected)
    end
  end

  describe 'mathematical properties' do
    it 'preserves tanh bounds' do
      neuron = Neuron.new(3)

      # Test with various input combinations
      test_inputs = [
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [-1.0, -1.0, -1.0],
        [10.0, 10.0, 10.0],
        [-10.0, -10.0, -10.0]
      ]

      test_inputs.each do |input|
        result = neuron.call(input)
        expect(result.data).to be_between(-1.0, 1.0)
      end
    end

    it 'is deterministic with same inputs and weights' do
      neuron = Neuron.new(2)

      # Set specific weights and bias
      neuron.instance_variable_set(:@w, [Value.new(0.5), Value.new(-0.3)])
      neuron.instance_variable_set(:@b, Value.new(0.1))

      input = [1.0, 2.0]
      result1 = neuron.call(input)
      result2 = neuron.call(input)

      expect(result1.data).to eq(result2.data)
    end
  end

  describe 'integration with Value class' do
    it 'works with Value arithmetic operations' do
      neuron = Neuron.new(2)
      input = [Value.new(1.0), Value.new(2.0)]
      result = neuron.call(input)

      # Should be able to perform operations with the result
      doubled = result + result
      expect(doubled).to be_a(Value)

      squared = result * result
      expect(squared).to be_a(Value)
    end

    it 'maintains computation graph for backpropagation' do
      neuron = Neuron.new(2)
      input = [Value.new(1.0), Value.new(2.0)]
      result = neuron.call(input)

      # The computation graph should be preserved
      expect(result).to respond_to(:left)
      expect(result).to respond_to(:op)
      expect(result.op).to eq(:tanh)
    end
  end
end
