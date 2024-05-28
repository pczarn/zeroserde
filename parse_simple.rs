

#[derive(Clone, Debug)]
enum Value {
    Digits(String),
    Float(f64),
    None,
}

fn calc(expr: &str) -> f64 {
    let mut grammar = Grammar::new();
    let sum = grammar.make_symbol("sum");
    let factor = grammar.make_symbol("factor");
    let op_mul = grammar.make_symbol("op_mul");
    let op_div = grammar.make_symbol("op_div");
    let lparen = grammar.make_symbol("lparen");
    let rparen = grammar.make_symbol("rparen");
    let expr_sym = grammar.make_symbol("expr_sym");
    let op_minus = grammar.make_symbol("op_minus");
    let op_plus = grammar.make_symbol("op_plus");
    let number = grammar.make_symbol("number");
    let whole = grammar.make_symbol("whole");
    let digit = grammar.make_symbol("digit");
    let dot = grammar.make_symbol("dot");
    // sum ::= sum [+-] factor
    // sum ::= factor
    // factor ::= factor [*/] expr
    // factor ::= expr
    // expr ::= '(' sum ')' | '-' expr | number
    // number ::= whole | whole '.' whole
    // whole ::= whole [0-9] | [0-9]
    grammar.rule(sum).rhs([sum, op_plus, factor]).id(0).build();
    grammar.rule(sum).rhs([sum, op_minus, factor]).id(1).build();
    grammar.rule(sum).rhs([factor]).id(2).build();
    grammar.rule(factor).rhs([factor, op_mul, expr_sym]).id(3).build();
    grammar.rule(factor).rhs([factor, op_div, expr_sym]).id(4).build();
    grammar.rule(factor).rhs([expr_sym]).id(5).build();
    grammar.rule(expr_sym).rhs([lparen, sum, rparen]).id(6).build();
    grammar.rule(expr_sym).rhs([op_minus, expr_sym]).id(7).build();
    grammar.rule(expr_sym).rhs([number]).id(8).build();
    grammar.rule(number).rhs([whole]).id(9).build();
    grammar.rule(number).rhs([whole, dot, whole]).id(10).build();
    grammar.rule(whole).rhs([whole, digit]).id(11).build();
    grammar.rule(whole).rhs([digit]).id(12).build();
    grammar.start_symbol(sum);
    let binarized_grammar = grammar.binarize();
    let mut recognizer = Recognizer::new(binarized_grammar);
    for (i, ch) in expr.chars().enumerate() {
        let terminal = match ch {
            '-' => op_minus,
            '.' => dot,
            '0' ..= '9' => digit,
            '(' => lparen,
            ')' => rparen,
            '*' => op_mul,
            '/' => op_div,
            '+' => op_plus,
            ' ' => continue,
            other => panic!("invalid character {}", other)
        };
        recognizer.begin_earleme();
        recognizer.scan(terminal, ch as u32);
        assert!(recognizer.end_earleme(), "parse failed at character {}", i);
    }
    let finished_node = recognizer.finished_node.expect("parse failed");
    let mut evaluator = Evaluator::new(
        |rule_id, args: &[Value]| {
            match (
                rule_id,
                args.get(0).cloned().unwrap_or(Value::None),
                args.get(1).cloned().unwrap_or(Value::None),
                args.get(2).cloned().unwrap_or(Value::None),
            ) {
                (0, Value::Float(left), _, Value::Float(right)) => {
                    Value::Float(left + right)
                }
                (1, Value::Float(left), _, Value::Float(right)) => {
                    Value::Float(left - right)
                }
                (2, val, Value::None, Value::None) => {
                    val
                }
                (3, Value::Float(left), _, Value::Float(right)) => {
                    Value::Float(left * right)
                }
                (4, Value::Float(left), _, Value::Float(right)) => {
                    Value::Float(left / right)
                }
                (5, val, Value::None, Value::None) => {
                    val
                }
                (6, _, val, _) => {
                    val
                }
                (7, _, Value::Float(num), Value::None) => {
                    Value::Float(-num)
                }
                (8, Value::Digits(digits), Value::None, Value::None) => {
                    Value::Float(digits.parse::<f64>().unwrap())
                }
                (9, val @ Value::Digits(..), _, _) => {
                    val
                }
                (10, Value::Digits(before_dot), _, Value::Digits(after_dot)) => {
                    let mut digits = before_dot;
                    digits.push('.');
                    digits.push_str(&after_dot[..]);
                    Value::Digits(digits)
                }
                (11, Value::Digits(mut num), Value::Digits(digit), _) => {
                    num.push_str(&digit[..]);
                    Value::Digits(num)
                }
                (12, val @ Value::Digits(..), _, _) => {
                    val
                }
                other => panic!("unknown rule id {:?} or args {:?}", rule_id, args)
            }
        },
        |terminal, values| {
            if terminal == digit {
                Value::Digits((values as u8 as char).to_string())
            } else {
                Value::None
            }
        }
    );
    let result = evaluator.evaluate(&mut recognizer.forest, finished_node);
    if let Value::Float(num) = result {
        num
    } else {
        panic!("evaluation failed")
    }
}