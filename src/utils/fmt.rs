pub fn unnecessary_brackets(result: &str) -> bool {
    if result.starts_with('(') && result.ends_with(')') {
        if result.split(')').count() == 2 {
            return true;
        }
        let mut count = 0;
        for c in result.chars() {
            if c == '(' {
                count += 1;
            } else if c == ')' {
                count -= 1;
            }
            if count >= 1 {
                return true;
            }
        }
    }
    false
}
