use std::collections::{BTreeMap, BTreeSet, BinaryHeap};
use std::mem;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
struct Symbol(u32);

struct Grammar<const S: usize> {
    rules: Vec<Rule>,
    start_symbol: Symbol,
    symbol_names: [&'static str; S],
    gen_symbol_lhs: Vec<Symbol>,
}

struct Rule {
    lhs: Symbol,
    rhs0: Symbol,
    rhs1: Option<Symbol>,
    id: Option<usize>,
}

struct Tables<const S: usize> {
    prediction_matrix: [[bool; S]; S],
    start_symbol: Symbol,
    rules: Vec<Rule>,
    completions: Vec<Vec<PredictionTransition>>,
    symbol_names: Vec<String>,
    gen_symbol_lhs: Vec<Symbol>,
}

#[derive(Copy, Clone, Debug)]
struct PredictionTransition {
    symbol: Symbol,
    dot: usize,
    is_unary: bool,
}

// Forest

struct Forest {
    graph: Vec<Node>,
    eval: Vec<Option<usize>>,
}

#[derive(Clone)]
enum Node {
    Product {
        action: u32,
        left_factor: NodeHandle,
        right_factor: Option<NodeHandle>,
    },
    Leaf {
        terminal: Symbol,
        values: u32,
    },
}

const NULL_ACTION: u32 = !0;

// Recognizer

struct Recognizer<const S: usize> {
    tables: Tables<S>,
    earley_chart: Vec<EarleySet<S>>,
    next_set: EarleySet<S>,
    complete: BinaryHeap<CompletedItem>,
    forest: Forest,
    finished_node: Option<NodeHandle>,
}

struct EarleySet<const S: usize> {
    predicted: [bool; S],
    medial: Vec<Item>,
}

#[derive(Ord, PartialOrd, Eq, PartialEq)]
struct Item {
    postdot: Symbol,
    dot: usize,
    origin: usize,
    node: NodeHandle,
}

#[derive(Clone, Copy, Ord, PartialOrd, Eq, PartialEq)]
struct CompletedItem {
    origin: usize,
    dot: usize,
    left_node: NodeHandle,
    right_node: Option<NodeHandle>,
}

#[derive(Eq, PartialEq, Ord, PartialOrd)]
enum MaybePostdot {
    Binary(Symbol),
    Unary,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
struct NodeHandle(usize);

impl Symbol {
    fn usize(self) -> usize {
        self.0 as usize
    }
}

impl<const S: usize> EarleySet<S> {
    fn new() -> Self {
        EarleySet {
            predicted: [false; S],
            medial: vec![],
        }
    }
}

impl<const S: usize> Grammar<S> {
    fn new(symbol_names: [&'static str; S], start_symbol: usize) -> Self {
        Self {
            rules: vec![],
            start_symbol: Symbol(start_symbol as u32),
            symbol_names,
            gen_symbol_lhs: vec![]
        }
    }

    fn rule<const N: usize>(&mut self, lhs: Symbol, mut rhs: [Symbol; N], id: usize) {
        let mut cur_rhs0 = rhs[0];
        for i in 1 .. N - 1 {
            self.rules.push(Rule {
                lhs: Symbolcur_rhs0 + 1,
                rhs0: cur_rhs0,
                rhs1: Some(rhs[i]),
                id: None,
            });
            cur_rhs0 = Symbol(self.gen_symbol_lhs.len() as u32);
            self.gen_symbol_lhs.push(cur_rhs0 + 1);
        }
        self.gen_symbol_lhs.push(lhs);
        self.rules.push(Rule {
            lhs,
            rhs0: cur_rhs0,
            rhs1: if N == 1 { None } else { Some(rhs[N - 1]) },
            id: Some(id),
        });
    }

    fn sort_rules(&mut self) {
        self.rules.sort_by(|a, b| a.lhs.cmp(&b.lhs));
    }
}

// Implementation for the recognizer.
//
// The recognizer has a chart of earley sets (Vec<EarleySet>) as well as the last set (next_set).
//
// A typical loop that utilizes the recognizer:
//
// - for character in string {
// 1.   recognizer.begin_earleme();
// 2.   recognizer.scan(token_to_symbol(character), values());
//        2a. complete
// 3.   recognizer.end_earleme();
//        3a. self.complete_all_sums_entirely();
//        3b. self.sort_medial_items();
//        3c. self.prediction_pass();
// - }
//
impl<const S: usize> Recognizer<S> {
    fn new(mut grammar: Grammar<S>) -> Self {
        grammar.sort_rules();
        let mut result = Self {
            tables: Tables::new(&grammar),
            earley_chart: vec![],
            next_set: EarleySet::new(),
            forest: Forest::new(&grammar),
            // complete: BinaryHeap::new_by_key(Box::new(|completed_item| (completed_item.origin, completed_item.dot))),
            complete: BinaryHeap::with_capacity(64),
            finished_node: None,
        };
        result.initialize();
        result
    }

    fn initialize(&mut self) {
        // self.earley_chart.push(EarleySet {
        //     predicted: self.tables.prediction_matrix[self.tables.start_symbol.usize()].clone(),
        //     medial: vec![],
        // });
        let es = EarleySet {
            predicted: self.tables.prediction_matrix[self.tables.start_symbol.usize()],
            medial: vec![],
        };
        // self.earley_chart.push(mem::replace(&mut self.next_set, EarleySet::new(self.tables.num_syms)));
        self.earley_chart.push(es);
    }

    fn begin_earleme(&mut self) {
        // nothing to do
    }

    fn scan(&mut self, terminal: Symbol, values: u32) {
        let earleme = self.earley_chart.len() - 1;
        let node = self.forest.leaf(terminal, earleme + 1, values);
        self.complete(earleme, terminal, node);
    }

    fn end_earleme(&mut self) -> bool {
        if self.is_exhausted() {
            false
        } else {
            // Completion pass, which saves successful parses.
            self.finished_node = None;
            self.complete_all_sums_entirely();
            // Do the rest.
            self.sort_medial_items();
            self.prediction_pass();
            self.earley_chart.push(mem::replace(
                &mut self.next_set,
                EarleySet::new(self.tables.num_syms),
            ));
            true
        }
    }

    fn is_exhausted(&self) -> bool {
        self.next_set.medial.len() == 0 && self.complete.is_empty()
    }

    fn complete_all_sums_entirely(&mut self) {
        while let Some(&ei) = self.complete.peek() {
            let lhs_sym = self.tables.get_lhs(ei.dot);
            while let Some(&ei2) = self.complete.peek() {
                if ei.origin == ei2.origin && lhs_sym == self.tables.get_lhs(ei2.dot) {
                    self.forest.push_summand(ei2);
                    self.complete.pop();
                } else {
                    break;
                }
            }
            let node = self.forest.sum(lhs_sym, ei.origin);
            if ei.origin == 0 && lhs_sym == self.tables.start_symbol {
                self.finished_node = Some(node);
            }
            self.complete(ei.origin, lhs_sym, node);
        }
    }

    /// Sorts medial items with deduplication.
    fn sort_medial_items(&mut self) {
        let tables = &self.tables;
        // Build index by postdot
        // These medial positions themselves are sorted by postdot symbol.
        self.next_set.medial.sort_unstable();
    }

    fn prediction_pass(&mut self) {
        // Iterate through medial items in the current set.
        let iter = self.next_set.medial.iter();
        // For each medial item in the current set, predict its postdot symbol.
        let destination = &mut self.next_set.predicted;
        for ei in iter {
            if let Some(postdot) = self
                .tables
                .get_rhs1(ei.dot)
                .filter(|postdot| !destination.contains(postdot.usize()))
            {
                // Prediction happens here. We would prefer to call `self.predict`, but we can't,
                // because `self.medial` is borrowed by `iter`.
                let source = &self.tables.prediction_matrix[postdot.usize()];
                destination.union_with(source);
            }
        }
    }

    fn complete(&mut self, earleme: usize, symbol: Symbol, node: NodeHandle) {
        if self.earley_chart[earleme]
            .predicted
            [symbol.usize()]
        {
            self.complete_medial_items(earleme, symbol, node);
            self.complete_predictions(earleme, symbol, node);
        }
    }

    fn complete_medial_items(&mut self, earleme: usize, symbol: Symbol, right_node: NodeHandle) {
        let inner_start = {
            // we use binary search to narrow down the range of items.
            let set_idx = self.earley_chart[earleme]
                .medial
                .binary_search_by(|ei| (self.tables.get_rhs1(ei.dot), 1).cmp(&(Some(symbol), 0)));
            match set_idx {
                Ok(idx) | Err(idx) => idx,
            }
        };

        let rhs1_eq = |ei| self.tables.get_rhs1(ei.dot) == Some(symbol);
        for item in self.earley_chart[earleme].medial[inner_start..]
            .iter()
            .take_while(rhs1_eq)
        {
            self.complete.push(CompletedItem {
                dot: item.dot,
                origin: item.origin,
                left_node: item.node,
                right_node: Some(right_node),
            });
        }
    }

    fn complete_predictions(&mut self, earleme: usize, symbol: Symbol, node: NodeHandle) {
        for trans in self.tables.unary_completions(symbol) {
            if self.earley_chart[earleme]
                .predicted
                .contains(trans.symbol.usize())
            {
                if trans.is_unary {
                    self.complete.push(CompletedItem {
                        origin: earleme,
                        dot: trans.dot,
                        left_node: node,
                        right_node: None,
                    });
                } else {
                    self.next_set.medial.push(Item {
                        origin: earleme,
                        dot: trans.dot,
                        node: node,
                    });
                }
            }
        }
    }

    // fn log_last_earley_set(&self) {
    //     let dots = self.dots_for_log(self.earley_chart.last().unwrap());
    //     for (rule_id, dots) in dots {
    //         print!("{} ::= ", self.tables.symbol_names[self.tables.get_lhs(rule_id).usize()]);
    //         if let Some(origins) = dots.get(&0) {
    //             print!("{:?}", origins);
    //         }
    //         print!(" {} ", self.tables.symbol_names[self.tables.get_rhs0(rule_id).unwrap().usize()]);
    //         if let Some(origins) = dots.get(&1) {
    //             print!("{:?}", origins);
    //         }
    //         if let Some(rhs1) = self.tables.get_rhs1(rule_id) {
    //             print!(" {} ", self.tables.symbol_names[rhs1.usize()]);
    //         }
    //         println!();
    //     }
    //     println!();
    // }

    // fn log_earley_set_diff(&self) {
    //     let dots_last_by_id = self.dots_for_log(self.earley_chart.last().unwrap());
    //     let mut dots_next_by_id = self.dots_for_log(&self.next_set);
    //     let mut rule_ids: BTreeSet<usize> = BTreeSet::new();
    //     rule_ids.extend(dots_last_by_id.keys());
    //     rule_ids.extend(dots_next_by_id.keys());
    //     for item in self.complete.iter() {
    //         let position = if self.tables.get_rhs1(item.dot).is_some() { 2 } else { 1 };
    //         dots_next_by_id.entry(item.dot).or_insert(BTreeMap::new()).entry(position).or_insert(BTreeSet::new()).insert(item.origin);
    //     }
    //     let mut empty_diff = true;
    //     for rule_id in rule_ids {
    //         let dots_last = dots_last_by_id.get(&rule_id);
    //         let dots_next = dots_next_by_id.get(&rule_id);
    //         if dots_last == dots_next {
    //             continue;
    //         }
    //         empty_diff = false;
    //         print!("from {} ::= ", self.tables.symbol_names[self.tables.get_lhs(rule_id).usize()]);
    //         if let Some(origins) = dots_last.and_then(|d| d.get(&0)) {
    //             print!("{:?}", origins);
    //         }
    //         print!(" {} ", self.tables.symbol_names[self.tables.get_rhs0(rule_id).unwrap().usize()]);
    //         if let Some(origins) = dots_last.and_then(|d| d.get(&1)) {
    //             print!("{:?}", origins);
    //         }
    //         if let Some(rhs1) = self.tables.get_rhs1(rule_id) {
    //             print!(" {} ", self.tables.symbol_names[rhs1.usize()]);
    //         }
    //         println!();
    //         print!("to   {} ::= ", self.tables.symbol_names[self.tables.get_lhs(rule_id).usize()]);
    //         if let Some(origins) = dots_next.and_then(|d| d.get(&0)) {
    //             print!("{:?}", origins);
    //         }
    //         print!(" {} ", self.tables.symbol_names[self.tables.get_rhs0(rule_id).unwrap().usize()]);
    //         if let Some(origins) = dots_next.and_then(|d| d.get(&1)) {
    //             print!("{:?}", origins);
    //         }
    //         if let Some(rhs1) = self.tables.get_rhs1(rule_id) {
    //             print!(" {} ", self.tables.symbol_names[rhs1.usize()]);
    //         }
    //         if let Some(origins) = dots_next.and_then(|d| d.get(&2)) {
    //             print!("{:?}", origins);
    //         }
    //         println!();
    //     }
    //     if empty_diff {
    //         println!("no diff");
    //         println!();
    //     } else {
    //         println!();
    //     }
    // }

    // fn dots_for_log(&self, es: &EarleySet) -> BTreeMap<usize, BTreeMap<usize, BTreeSet<usize>>> {
    //     let mut dots = BTreeMap::new();
    //     for (i, rule) in self.tables.rules.iter().enumerate() {
    //         if es.predicted.contains(rule.lhs.usize()) {
    //             dots.entry(i).or_insert(BTreeMap::new()).entry(0).or_insert(BTreeSet::new()).insert(self.earleme());
    //         }
    //     }
    //     for item in &es.medial {
    //         dots.entry(item.dot).or_insert(BTreeMap::new()).entry(1).or_insert(BTreeSet::new()).insert(item.origin);
    //     }
    //     dots
    // }
}

impl<const S: usize> Tables<S> {
    fn new(grammar: &Grammar<S>) -> Self {
        let mut result = Self {
            prediction_matrix: [[false; S]; S],
            start_symbol: grammar.start_symbol,
            rules: vec![],
            completions: vec![],
            symbol_names: grammar.symbol_names,
        };
        result.populate(&grammar);
        result
    }

    fn populate(&mut self, grammar: &BinarizedGrammar) {
        self.populate_prediction_matrix(grammar);
        self.populate_rules(grammar);
        self.populate_completions(grammar);
    }

    fn populate_prediction_matrix(&mut self, grammar: &BinarizedGrammar) {
        for rule in &grammar.rules {
            self.prediction_matrix[rule.lhs.usize()][rule.rhs0.usize()] = true;
        }
        self.reflexive_closure();
        self.transitive_closure();
    }

    fn reflexive_closure(&mut self) {
        for i in 0..self.num_syms {
            self.prediction_matrix[i][i] = true;
        }
    }

    fn transitive_closure(&mut self) {
        for pos in 0..self.num_syms {
            let (rows0, rows1) = self.prediction_matrix.split_at_mut(pos);
            let (rows1, rows2) = rows1.split_at_mut(1);
            for dst_row in rows0.iter_mut().chain(rows2.iter_mut()) {
                if dst_row[pos] {
                    dst_row.union_with(&rows1[0]);
                }
            }
        }
    }

    fn populate_rules(&mut self, grammar: &BinarizedGrammar) {
        self.rules = grammar.rules.clone();
    }

    fn populate_completions(&mut self, grammar: &BinarizedGrammar) {
        self.completions.resize(self.num_syms, vec![]);
        for (i, rule) in grammar.rules.iter().enumerate() {
            self.completions[rule.rhs0.usize()].push(PredictionTransition {
                symbol: rule.lhs,
                dot: i,
                is_unary: rule.rhs1.is_none(),
            });
        }
    }

    fn get_rhs0(&self, n: usize) -> Option<Symbol> {
        self.rules.get(n).map(|rule| rule.rhs0)
    }

    fn get_rhs1(&self, n: usize) -> Option<Symbol> {
        self.rules.get(n).and_then(|rule| rule.rhs1)
    }

    fn get_rhs1_cmp(&self, dot: usize) -> MaybePostdot {
        match self.rules[dot].rhs1 {
            None => MaybePostdot::Unary,
            Some(rhs1) => MaybePostdot::Binary(rhs1),
        }
    }

    fn get_lhs(&self, n: usize) -> Symbol {
        self.rules[n].lhs
    }
}

impl Forest {
    fn new<const S: usize>(grammar: &Grammar<S>) -> Self {
        Self {
            graph: vec![],
            eval: grammar.rules.iter().map(|rule| rule.id).collect(),
        }
    }

    fn leaf(&mut self, terminal: Symbol, _x: usize, values: u32) -> NodeHandle {
        let handle = NodeHandle(self.graph.len());
        self.graph.push(Node::Leaf { terminal, values });
        handle
    }

    fn push_summand(&mut self, item: CompletedItem) -> NodeHandle {
        let handle = NodeHandle(self.graph.len());
        let eval = self.eval[item.dot].map(|id| id as u32);
        self.graph.push(Node::Product {
            action: eval.unwrap_or(NULL_ACTION),
            left_factor: item.left_node,
            right_factor: item.right_node,
        });
        handle
    }
}

struct Evaluator<F, G> {
    eval_product: F,
    eval_leaf: G,
}

impl<T, F, G> Evaluator<F, G>
where
    F: Fn(u32, &[T]) -> T + Copy,
    G: Fn(Symbol, u32) -> T + Copy,
    T: Clone + ::std::fmt::Debug,
{
    fn new(eval_product: F, eval_leaf: G) -> Self {
        Self {
            eval_product,
            eval_leaf,
        }
    }

    fn evaluate(&mut self, forest: &mut Forest, finished_node: NodeHandle) -> T {
        self.evaluate_rec(forest, finished_node)[0].clone()
    }

    fn evaluate_rec(&mut self, forest: &mut Forest, handle: NodeHandle) -> Vec<T> {
        match forest.graph[handle.0] {
            Node::Product { left_factor, right_factor, action } => {
                let mut result = self.evaluate_rec(forest, left_factor);
                if let Some(factor) = right_factor {
                    let v = self.evaluate_rec(forest, factor);
                    result.extend(v);
                }
                if action != NULL_ACTION {
                    vec![(self.eval_product)(action as u32, &result[..])]
                } else {
                    result
                }
            }
            Node::Leaf { terminal, values } => {
                vec![(self.eval_leaf)(terminal, values)]
            }
        }
    }
}
