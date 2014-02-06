# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

#default_value:foo
#include: other.manifest
#
#[test_name.js]
#  expected: ERROR
#
#  [subtest 1]
#    expected:
#      os == win: FAIL #This is a comment
#      PASS
#

import types
from cStringIO import StringIO
import operator

class ParseError(Exception):
    pass

eol = object
group_start = object
group_end = object
digits = "0123456789"
open_parens = "[("
close_parens = "])"
parens = open_parens + close_parens
operator_chars = "="

unary_operators = ["not"]
binary_operators = ["==", "and", "or"]

operators = ["==", "not", "and", "or"]

class TokenTypes(object):
    def __init__(self):
        for type in ["group_start", "group_end", "paren", "separator", "ident", "string", "number", "eof"]:
            setattr(self, type, type)

token_types = TokenTypes()

class Tokenizer(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.indent_levels = [0]
        self.state = self.line_start_state
        self.next_state = self.data_line_state
        self.line_number = 0

    def tokenize(self, stream):
        self.reset()
        if type(stream) in types.StringTypes:
            stream = StringIO(stream)

        for i, line in enumerate(stream):
            self.state = self.line_start_state
            self.line_number = i + 1
            self.index = 0
            self.line = line.rstrip()
            if self.line:
                while self.state != self.eol_state:
                    tokens = self.state()
                    if tokens:
                        for token in tokens:
                            yield token
        while True:
            yield (token_types.eof, None)

    def char(self):
        if self.index == len(self.line):
            return eol
        return self.line[self.index]

    def consume(self):
        if self.index < len(self.line):
            self.index += 1

    def peek(self, length):
        return self.line[self.index:self.index + length]

    def skip_whitespace(self):
        while self.char() == " ":
            self.consume()

    def eol_state(self):
        pass

    def line_start_state(self):
        self.skip_whitespace()
        assert self.char() != eol
        if self.index > self.indent_levels[-1]:
            self.indent_levels.append(self.index)
            yield (token_types.group_start, None)
        else:
            while self.index < self.indent_levels[-1]:
                self.indent_levels.pop()
                yield (token_types.group_end, None)
                #This is terrible; if we were parsing an expression
                #then the next_state will be expr_or_value but when we deindent
                #it must always be a heading or key next so we go back to data_line_state
                self.next_state = self.data_line_state
            if self.index != self.indent_levels[-1]:
                raise ParseError("Unexpected indent")

        self.state = self.next_state

    def data_line_state(self):
        if self.char() == "[":
            yield (token_types.paren, self.char())
            self.consume()
            self.state = self.heading_state
        else:
            self.state = self.key_state

    def heading_state(self):
        index_0 = self.index
        skip_indexes = []
        while True:
            c = self.char()
            if c == "\\":
                self.consume()
                c = self.char()
                if c == eol:
                    raise ParseError("Unexpected EOL in heading")
                elif c == "]":
                    skip_indexes.append(self.index-1)
            elif c == "]":
                break
            elif c == eol:
                raise ParseError("EOL in heading")
            else:
                self.consume()

        self.state = self.line_end_state
        index_1 = self.index
        parts = []
        min_index = index_0
        for index in skip_indexes:
            parts.append(self.line[min_index:index])
            min_index = index + 1
        parts.append(self.line[min_index:index_1])
        yield (token_types.string, "".join(parts))
        yield (token_types.paren, "]")
        self.consume()
        self.state = self.line_end_state
        self.next_state = self.data_line_state

    def key_state(self):
        index_0 = self.index
        while True:
            c = self.char()
            if c == " ":
                index_1 = self.index
                self.skip_whitespace()
                if self.char() != ":":
                    raise ParseError("Space in key name")
                break
            elif c == ":":
                index_1 = self.index
                break
            elif c == eol:
                raise ParseError("EOL in key name (missing ':'?)")
            else:
                self.consume()
        yield (token_types.string, self.line[index_0:index_1])
        yield (token_types.separator, ":")
        self.consume()
        self.state = self.after_key_state

    def after_key_state(self):
        self.skip_whitespace()
        c = self.char()
        if c == "#":
            self.next_state = self.expr_or_value_state
            self.state = self.comment_state
        elif c == eol:
            self.next_state = self.expr_or_value_state
            self.state = self.eol_state
        else:
            self.state = self.value_state

    def value_state(self):
        index_0 = self.index
        if self.char() in ("'", '"'):
            quote_char = self.char()
            self.consume()
            yield (token_types.string, self.read_string(quote_char))
        else:
            index_1 = self.index
            while True:
                c = self.char()
                if c == "\\":
                    self.consume()
                    if self.char() == eol:
                        raise ParseError("EOL in character escape")
                elif c == "#":
                    self.state = self.comment_state
                    break
                elif c == " ":
                    #prevent whitespace before comments from being included in the value
                    pass
                elif c == eol:
                    break
                else:
                    index_1 = self.index
                self.consume()
            yield (token_types.string, self.line[index_0:index_1 + 1])
        self.state = self.line_end_state

    def comment_state(self):
        self.state = self.eol_state

    def line_end_state(self):
        self.skip_whitespace()
        c = self.char()
        if c == "#":
            self.state = self.comment_state
        elif c == eol:
            self.state = self.eol_state
        else:
            raise ParseError("Junk before EOL c")

    def read_string(self, quote_char):
        index_0 = self.index
        while True:
            c = self.char()
            if c == "\\":
                self.consume()
                if self.char == eol:
                    raise ParseError("EOL following quote")
                self.consume()
            elif c == quote_char:
                break
            elif c == eol:
                raise ParseError("EOL in quoted string")
            else:
                self.consume()
        rv = self.line[index_0:self.index]
        self.consume()
        return rv

    def expr_or_value_state(self):
        if self.peek(3) == "if ":
            self.state = self.expr_state
        else:
            self.state = self.value_state

    def expr_state(self):
        self.skip_whitespace()
        c = self.char()
        if c == eol:
            raise ParseError("EOL in expression")
        elif c in "'\"":
            self.consume()
            yield (token_types.string, self.read_string(c))
        elif c == "#":
            raise ParseError("Comment before end of expression")
        elif c == ":":
            yield (token_types.separator, c)
            self.consume()
            self.state = self.value_state
        elif c in parens:
            self.consume()
            yield (token_types.paren, c)
        elif c in operators:
            self.state = self.operator_state
        elif c in digits:
            self.state = self.digit_state
        else:
            self.state = self.ident_state

    def operator_state(self):
        #Only symbolic operators
        index_0 = self.index
        while True:
            c = self.char()
            if c == eol:
                break
            elif c in operator_chars:
                self.consume()
            else:
                self.state = self.expr_state
                break
        yield (token_types.ident, token)

    def digit_state(self):
        index_0 = self.index
        seen_dot = False
        while True:
            c = self.char()
            if c == eol:
                break
            elif c in digits:
                self.consume()
            elif c == ".":
                if seen_dot:
                    raise ParseError("Invalid number")
                self.consume()
                seen_dot = True
            elif c in parens:
                break
            elif c in operators:
                break
            elif c == " ":
                break
            elif c == ":":
                break
            else:
                raise ParseError("Invalid character in number")

        self.state = self.expr_state
        yield (token_types.number, self.line[index_0:self.index])

    def ident_state(self):
        index_0 = self.index
        while True:
            c = self.char()
            if c == eol:
                break
            elif c == ".":
                break
            elif c in parens:
                break
            elif c in operators:
                break
            elif c == " ":
                break
            elif c == ":":
                break
            else:
                self.consume()
        self.state = self.expr_state
        yield (token_types.ident, self.line[index_0:self.index])


class Parser(object):
    def __init__(self):
        self.reset()
    def reset(self):
        self.token = None
        self.unary_operators = "!"
        self.binary_operators = frozenset(["&&", "||", "=="])
        self.tokenizer = Tokenizer()
        self.token_generator = None
        self.tree = Treebuilder(DataNode(None))
        self.expr_builder = None
        self.expr_builders = []

    def parse(self, input):
        self.reset()
        self.token_generator = self.tokenizer.tokenize(input)
        self.consume()
        self.manifest()
        return self.tree.node

    def consume(self):
        self.token = self.token_generator.next()

    def expect(self, type, value=None):
        if self.token[0] != type:
            raise ParseError
        if value is not None:
            if self.token[1] != value:
                raise ParseError

        self.consume()

    def manifest(self):
        self.data_block()
        self.expect(token_types.eof)

    def data_block(self):
        while self.token[0] == token_types.string:
            self.tree.append(KeyValueNode(self.token[1]))
            self.consume()
            self.expect(token_types.separator)
            self.value_block()
            self.tree.pop()

        while self.token == (token_types.paren, "["):
            self.consume()
            if self.token[0] != token_types.string:
                raise ParseError
            self.tree.append(DataNode(self.token[1]))
            self.consume()
            self.expect(token_types.paren, "]")
            if self.token[0] == token_types.group_start:
                self.consume()
                self.data_block()
                self.eof_or_end_group()
            self.tree.pop()

    def eof_or_end_group(self):
        if self.token[0] != token_types.eof:
            self.expect(token_types.group_end)

    def value_block(self):
        if self.token[0] == token_types.string:
            self.value()
        elif self.token[0] == token_types.group_start:
            self.consume()
            self.expression_values()
            if self.token[0] == token_types.string:
                self.value()
            self.eof_or_end_group()
        else:
            raise ParseError

    def expression_values(self):
        while self.token == (token_types.ident, "if"):
            self.consume()
            self.tree.append(ConditionalNode())
            self.expr_start()
            self.expect(token_types.separator)
            if self.token[0] == token_types.string:
                self.value()
            else:
                raise ParseError
            self.tree.pop()

    def value(self):
        self.tree.append(ValueNode(self.token[1]))
        self.consume()
        self.tree.pop()

    def expr_start(self):
        old_tree = self.tree
        self.expr_builder = ExpressionBuilder()
        self.expr_builders.append(self.expr_builder)
        self.expr()
        expression = self.expr_builder.finish()
        self.expr_builders.pop()
        self.expr_builder = self.expr_builders[-1] if self.expr_builders else None
        if self.expr_builder:
            self.expr_builder.operands[-1].children[-1].append(expression)
        else:
            self.tree.append(expression)
            self.tree.pop()

    def expr(self):
        self.expr_operand()
        while (self.token[0] == token_types.ident and self.token[1] in binary_operators):
            self.expr_bin_op()
            self.expr_operand()

    def expr_operand(self):
        if self.token == (token_types.paren, "("):
            self.consume()
            self.expr_builder.push_operator(None)
            self.expr()
            self.expect(token_types.paren, ")")
            self.expr_builder.pop_operator()
        elif self.token[0] == token_types.ident and self.token[1] in unary_operators:
            self.expr_unary_op()
            self.expr_operand()
        elif self.token[0] in [token_types.string, token_types.ident]:
            self.expr_value()
        elif self.token[0] == token_types.number:
            self.expr_number()
        else:
            raise ParseError

    def expr_unary_op(self):
        if self.token[1] in unary_operators:
            self.expr_builder.push_operator(UnaryOperatorNode(self.token[1]))
            self.consume()
        else:
            raise ParseError()

    def expr_bin_op(self):
        if self.token[1] in binary_operators:
            self.expr_builder.push_operator(BinaryOperatorNode(self.token[1]))
            self.consume()
        else:
            raise ParseError()

    def expr_value(self):
        node_type = {token_types.string: StringNode,
                     token_types.ident: VariableNode}[self.token[0]]
        self.expr_builder.push_operand(node_type(self.token[1]))
        self.consume()
        if self.token == (token_types.paren, "["):
            self.consume()
            self.expr_builder.operands[-1].append(IndexNode())
            self.expr_start()
            self.expect(token_types.paren, "]")

    def expr_number(self):
        self.expr_builder.push_operand(NumberNode(self.token[1]))
        self.consume()

class Treebuilder(object):
    def __init__(self, root):
        self.root = root
        self.node = root

    def append(self, node):
        self.node.append(node)
        self.node = node
        return node

    def pop(self):
        node = self.node
        self.node = self.node.parent
        return node

class Node(object):
    def __init__(self, data=None):
        self.data = data
        self.parent = None
        self.children = []

    def append(self, other):
        other.parent = self
        self.children.append(other)

    def __repr__(self):
        return "<%s %s>" % (self.__class__.__name__, self.data)

    def __str__(self):
        rv = [repr(self)]
        for item in self.children:
            rv.extend("  %s"%line for line in str(item).split("\n"))
        return "\n".join(rv)

class DataNode(Node):
    pass

class KeyValueNode(Node):
    pass

class ValueNode(Node):
    def append(self, other):
        raise TypeError

class ConditionalNode(Node):
    pass

class UnaryExpressionNode(Node):
    def __init__(self, operator, operand):
        Node.__init__(self)
        self.append(operator)
        self.append(operand)

    def append(self, other):
        Node.append(self, other)
        assert len(self.children) <= 2

class BinaryExpressionNode(Node):
    def __init__(self, operator, operand_0, operand_1):
        Node.__init__(self)
        self.append(operator)
        self.append(operand_0)
        self.append(operand_1)

    def append(self, other):
        Node.append(self, other)
        assert len(self.children) <= 3

class UnaryOperatorNode(Node):
    def append(self, other):
        raise TypeError

class BinaryOperatorNode(Node):
    def append(self, other):
        raise TypeError

class IndexNode(Node):
    pass

class VariableNode(Node):
    pass

class StringNode(Node):
    pass

class NumberNode(ValueNode):
    pass

class ExpressionBuilder(object):
    def __init__(self):
        self.operands = []
        self.operators = [None]

    def finish(self):
        while self.operators[-1] is not None:
            self.pop_operator()
        rv = self.pop_operand()
        assert self.is_empty()
        return rv

    def push_operator(self, operator):
        while self.precedence(self.operators[-1]) > self.precedence(operator):
            self.pop_operator()

        self.operators.append(operator)

    def pop_operator(self):
        operator = self.operators.pop()
        if isinstance(operator, BinaryOperatorNode):
            operand_1 = self.operands.pop()
            operand_0 = self.operands.pop()
            self.operands.append(BinaryExpressionNode(operator, operand_0, operand_1))
        else:
            operand_0 = self.operands.pop()
            self.operands.append(UnaryExpressionNode(operator, operand_0))

    def push_operand(self, node):
        self.operands.append(node)

    def pop_operand(self):
        return self.operands.pop()

    def is_empty(self):
        return len(self.operands) == 0 and all(item is None for item in self.operators)

    def precedence(self, operator):
        if operator is None:
            return 0
        return len(operators) - operators.index(operator.data)


class NodeVisitor(object):
    def visit(self, node):
        #This is ugly as hell, but we don't have multimethods and
        #they aren't trivial to fake without access to the class
        #object from the class body
        func = getattr(self, "visit_%s" % (node.__class__.__name__))
        return func(node)

class ManifestCompiler(NodeVisitor):
    def compile(self, tree, expr_data):
        self.expr_data = expr_data
        #Create a root node with the passed in expression data
        self.manifest = ManifestItem(None)
        self.manifest.update(expr_data)
        self.visit(tree)
        rv = self.manifest.children[0]
        rv.parent = None
        return rv

    def visit_DataNode(self, node):
        self.manifest = self.manifest.append(ManifestItem(node.data))

        for child in node.children:
            self.visit(child)

        self.manifest = self.manifest.parent

    def visit_KeyValueNode(self, node):
        key_name = node.data
        key_value = None
        for child in node.children:
            value = self.visit(child)
            if value is not None:
                key_value = value
                break
        if key_value is not None:
            self.manifest[key_name] = key_value

    def visit_ValueNode(self, node):
        return node.data

    def visit_ConditionalNode(self, node):
        assert len(node.children) == 2
        if self.visit(node.children[0]):
            return self.visit(node.children[1])

    def visit_StringNode(self, node):
        value = node.data
        for child in node.children:
            value = self.visit(child)(value)
        return value

    def visit_NumberNode(self, node):
        if "." in node.data:
            return float(node.data)
        else:
            return int(node.data)

    def visit_VariableNode(self, node):
        value = self.manifest[node.data]
        for child in node.children:
            value = self.visit(child)(value)
        return value

    def visit_IndexNode(self, node):
        assert len(node.children) == 1
        index = self.visit(node.children[0])
        return lambda x:x[index]

    def visit_UnaryExpressionNode(self, node):
        assert len(node.children) == 2
        operator = self.visit(node.children[0])
        operand = self.visit(node.children[1])

        return operator(operand)

    def visit_BinaryExpressionNode(self, node):
        assert len(node.children) == 3
        operator = self.visit(node.children[0])
        operand_0 = self.visit(node.children[1])
        operand_1 = self.visit(node.children[2])

        return operator(operand_0, operand_1)

    def visit_UnaryOperatorNode(self, node):
        return {"not": operator.not_}[node.data]

    def visit_BinaryOperatorNode(self, node):
        return {"and": operator.and_,
                "or": operator.or_,
                "==": operator.eq}[node.data]

class ManifestItem(object):
    def __init__(self, name):
        self.name = name
        self.children = []
        self._data = {}
        self.parent = None

    def __repr__(self):
        return "<ManifestItem %s>" %(self.name)

    def __str__(self):
        rv = [repr(self)]
        for item in self.children:
            rv.extend("  %s"%line for line in str(item).split("\n"))
        return "\n".join(rv)

    def __contains__(self, key):
        node = self
        while node is not None:
            if name in node._data:
                return True
            node = node.parent
        return False

    def __getitem__(self, name):
        node = self
        while node is not None:
            if name in node._data:
                return node._data[name]
            node = node.parent
        raise KeyError

    def __setitem__(self, name, value):
        self._data[name] = value

    def __delitem__(self, name):
        del self._data[name]

    def _flatten(self):
        rv = {}
        node = self
        while node is not None:
            for name, value in node._data.iteritems():
                if name not in rv:
                    rv[name] = value
            node = node.parent
        return rv

    def update(self, other):
        self._data.update(other)

    def iteritems(self):
        for item in self._flatten().iteritems():
            yield item

    def iterkeys(self):
        for item in self._flatten().iterkeys():
            yield item

    def itervalues(self):
        for item in self._flatten().itervalues():
            yield item

    def append(self, child):
        child.parent = self
        self.children.append(child)
        return child


def to_test_manifest(manifest):
    rv = []
    for test in manifest.children:
        test_data = dict(test.iteritems())
        test_data.update({"id":test.name, "subtests":[]})
        rv.append(test_data)
        for subtest in test.children:
            assert not subtest.children
            subtest_data = dict(subtest.iteritems())
            subtest_data.update({"id":subtest.name})
            test_data["subtests"].append(subtest_data)
    return rv

def compile(stream, expr_data):
    p = Parser()
    tree = p.parse(stream)
    c = ManifestCompiler()
    return c.compile(tree, expr_data)
