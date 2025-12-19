pub fn unnecessary_brackets(result: &str) -> bool {
    let bytes = result.as_bytes();
    if bytes.len() < 2 || bytes.first() != Some(&b'(') || bytes.last() != Some(&b')') {
        return false;
    }

    let mut depth: i32 = 0;
    for (idx, byte) in bytes.iter().enumerate() {
        match byte {
            b'(' => depth += 1,
            b')' => {
                if depth == 0 {
                    return false;
                }
                depth -= 1;
            }
            _ => {}
        }

        if depth == 0 && idx < bytes.len() - 1 {
            return false;
        }
    }

    depth == 0
}
