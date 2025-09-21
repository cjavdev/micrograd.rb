# frozen_string_literal: true

require 'spec_helper'
require_relative '../lib/value'

RSpec.describe Value do
  describe '#initialize' do
    it 'creates a Value with the given data' do
      value = Value.new(5.0)
      expect(value.data).to eq(5.0)
    end

    it 'converts string input to float' do
      value = Value.new("3.14")
      expect(value.data).to eq(3.14)
    end

    it 'converts integer input to float' do
      value = Value.new(10)
      expect(value.data).to eq(10.0)
    end
  end

  describe '#+' do
    it 'adds two values' do
      a = Value.new(5.0)
      b = Value.new(3.0)
      result = a + b
      expect(result.data).to eq(8.0)
    end

    it 'adds negative values' do
      a = Value.new(5.0)
      b = Value.new(-3.0)
      result = a + b
      expect(result.data).to eq(2.0)
    end

    it 'applies the chain rule to the _backward function' do
      a = Value.new(5.0)
      a.grad = 3.0
      b = Value.new(3.0)
      b.grad = 4.0
      result = a + b
      expect(result.data).to eq(8.0)
      expect(result._backward).to be_a(Proc)
      result.grad = 1.0
      result._backward.call
      expect(a.grad).to eq(4.0)
      expect(b.grad).to eq(5.0)
    end
  end

  describe '#*' do
    it 'multiplies two values' do
      a = Value.new(5.0)
      b = Value.new(3.0)
      result = a * b
      expect(result.data).to eq(15.0)
    end

    it "applies the chain rule to the _backward function" do
      a = Value.new(5.0)
      a.grad = 2.0
      b = Value.new(3.0)
      b.grad = 4.0
      result = a * b
      expect(result.data).to eq(15.0)
      expect(result._backward).to be_a(Proc)
      result.grad = 1.0
      result._backward.call
      expect(a.grad).to eq(2.0 + 3.0)
      expect(b.grad).to eq(4.0 + 5.0)
    end
  end

  describe '#**' do
    it 'raises a value to the power of another' do
      a = Value.new(5.0)
      result = a ** 3.0
      expect(result.data).to eq(125.0)
    end

    it "applies the chain rule to the _backward function" do
      a = Value.new(5.0)
      a.grad = 2.0
      result = a ** 2.0
      expect(result.data).to eq(25.0)
      expect(result._backward).to be_a(Proc)
      result.grad = 1.0
      result._backward.call
      expect(a.grad).to eq((2.0 * (5.0 ** 1.0) * 1.0) + 2.0)
    end
  end

  describe '#tanh' do
    it 'applies the hyperbolic tangent function to a value' do
      a = Value.new(5.0)
      result = a.tanh
      expect(result.data).to eq(0.9999092042625951)
    end

    it "applies the chain rule to the _backward function" do
      a = Value.new(5.0)
      a.grad = 2.0
      result = a.tanh
      expect(result.data).to eq(0.9999092042625951)
      expect(result._backward).to be_a(Proc)
      result.grad = 1.0
      result._backward.call
      expect(a.grad).to eq((1.0 - (0.9999092042625951 ** 2.0)) * 1.0 + 2.0)
    end
  end

  describe '#exp' do
    it 'applies the exponential function to a value' do
      a = Value.new(5.0)
      result = a.exp
      expect(result.data).to eq(148.4131591025766)
    end

    it "applies the chain rule to the _backward function" do
      a = Value.new(5.0)
      a.grad = 2.0
      result = a.exp
      expect(result.data).to eq(148.4131591025766)
      expect(result._backward).to be_a(Proc)
      result.grad = 1.0
      result._backward.call
      expect(a.grad).to eq(148.4131591025766 * 1.0 + 2.0)
    end
  end
end
