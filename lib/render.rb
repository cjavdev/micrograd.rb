require 'ruby-graphviz'
require 'set'

class Render
  def initialize(value)
    @value = value
  end

  def trace(x)
    nodes = Set.new
    edges = Set.new

    build = lambda do |v|
      unless nodes.include?(v)
        nodes.add(v)
        [v&.left, v&.right].compact.each do |child|
          edges.add([child, v])
          build.call(child)
        end
      end
    end

    build.call(x)
    [nodes, edges]
  end

  def render
    dot = GraphViz.new(:G, type: :digraph, rankdir: 'LR')
    nodes, edges = trace(@value)

    nodes.each do |n|
      uid = n.object_id.to_s
      label = "#{n.label} | data: #{n.data} | grad: #{n.grad}"
      dot.add_nodes(uid, label: label, shape: 'record')

      if !n.op.empty?
        op_uid = uid + n.op.to_s
        dot.add_nodes(op_uid, label: n.op.to_s)
        dot.add_edges(op_uid, uid)
      end
    end

    edges.each do |a, b|
      a_uid = a.object_id.to_s
      b_uid = b.object_id.to_s
      target_uid = b.op.empty? ? b_uid : b_uid + b.op.to_s
      dot.add_edges(a_uid, target_uid)
    end

    dot.output(png: 'graph.png')
  end
end