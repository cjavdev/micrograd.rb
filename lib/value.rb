# frozen_string_literal: true

require 'set'

class Value
  attr_reader :left, :right, :op
  attr_accessor :grad, :label, :_backward, :data

  def initialize(data, left: nil, right: nil, op: '', label: '')
    @data = data.to_f
    @grad = 0.0
    @left = left
    @right = right
    @op = op
    @label = label == '' ? ('A'..'Z').to_a.sample(3).join : label
    @_backward = -> do
      if !@left.nil? || !@right.nil?
        raise ArgumentError, "Non leaf node #{@label} #{@op} with children: l: #{@left.inspect} and r: #{@right.inspect} must have backward function"
      end
    end
  end

  def backward
    # Build topological order
    topo = []
    visited = Set.new

    def build_topo(v, visited, topo)
      if !visited.include?(v)
        visited.add(v)
        [v&.left, v&.right].compact.each { |child| build_topo(child, visited, topo) }
        topo << v
      end
    end

    build_topo(self, visited, topo)

    # Go one variable at a time and apply the chain rule to get its gradient
    @grad = 1.0

    topo.reverse_each { |v| v._backward&.call }
  end

  # Addition
  def +(other)
    other = other.is_a?(Value) ? other : Value.new(other)
    out = Value.new(@data + other.data, left: self, right: other, op: :+)

    # Store the backward function on the result, capturing the operands
    out._backward = -> do
      # byebug
      # puts "addition backward: #{@label}"
      self.grad += out.grad
      other.grad += out.grad
    end

    out
  end

  # Subtraction
  def -(other)
    other = other.is_a?(Value) ? other : Value.new(other)
    self + (-other)
  end

  # Multiplication
  def *(other)
    other = other.is_a?(Value) ? other : Value.new(other)
    out = Value.new(@data * other.data, left: self, right: other, op: :*)

    # Store the backward function on the result, capturing the operands
    out._backward = -> do
      # byebug
      # puts "multiplication backward: #{@label}"
      self.grad += other.data * out.grad
      other.grad += self.data * out.grad
    end

    out
  end

  # Division
  def /(other)
    other = other.is_a?(Value) ? other : Value.new(other)
    raise ZeroDivisionError, "Cannot divide by zero" if other.data == 0.0
    self * (other ** -1)
  end

  # Power operation
  def **(other)
    # Other must be a numeric
    raise ArgumentError, "Power must be a numeric" unless other.is_a?(Numeric)
    out = Value.new(@data ** other, left: self, right: nil, op: :**)

    # Store the backward function on the result, capturing the operands
    out._backward = -> do
      # byebug
      # puts "power backward: #{@label}"
      self.grad += other * (self.data ** (other - 1)) * out.grad
    end

    out
  end

  # Tangent hyperbolic
  def tanh
    out = Value.new(Math.tanh(@data), left: self, op: :tanh)

    # Store the backward function on the result, capturing the operand
    out._backward = -> do
      # puts "tanh backward: #{@label}"
      self.grad += (1 - (out.data ** 2)) * out.grad
    end

    out
  end

  # Exponential function
  def exp
    out = Value.new(Math.exp(@data), left: self, op: :exp)

    # Store the backward function on the result, capturing the operand
    out._backward = -> do
      # puts "exp backward: #{@label}"
      self.grad += out.data * out.grad
    end

    out
  end

  # String representation
  def to_s
    @data.to_s
  end

  def inspect
    "Value(#{@data}, l: #{@left}, r: #{@right}, op: #{@op}, label: #{@label})"
  end

  def -@
    self * -1
  end

  # Coercion method to handle operations with numeric types
  def coerce(other)
    [Value.new(other), self]
  end
end
