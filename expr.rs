use std::error::Error;

use crate::to_from::{lexer::Lexer, parser::{Node, Parser}};

pub struct Expression {
    tree: Node,

}

impl Expression {
    pub fn parse(input: String, indp: Option<char>) -> Result<Self, Rc<dyn Error>> {
        let mut lexer = Lexer::new(input, indp);
        let lexed = lexer.lex()?;
        let mut parser = Parser::new(lexed);
        let tree = parser.parse()?;
        Ok(Self { tree })
    }
}