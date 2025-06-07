
struct EditTable<'a, T: PartialEq<T>> {
    cells: Box<[usize]>,
    old: &'a [T],
    new: &'a [T],
    width: usize,
    height: usize,
}

impl<'a, T: PartialEq<T>> EditTable<'a, T> {
    fn new(old: &'a [T], new: &'a [T]) -> Self {
        let width = old.len() + 1;
        let height = new.len() + 1;
        let cell_count = width * height;
        Self {
            cells: Box::from_iter((0..cell_count).map(|_| 0)),
            old,
            new,
            width,
            height,
        }
    }

    fn compare(&self, x: usize, y: usize) -> bool {
        match (x, y) {
            (0, 0) => true,
            (0, _) => false,
            (_, 0) => false,
            (x, y) => {
                self.old[x - 1] == self.new[y - 1]
            }
        }
    }

    fn index_at(&self, x: usize, y: usize) -> usize {
        debug_assert!(x < self.width && y < self.height);
        y * self.width + x
    }

    fn get(&self, x: usize, y: usize) -> usize {
        let index = self.index_at(x, y);
        self.cells[index]
    }

    fn set(&mut self, x: usize, y: usize, value: usize) {
        let index = self.index_at(x, y);
        self.cells[index] = value;
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EditType {
    Deleted,
    Inserted,
    Unchanged,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Edit {
    /// usize is index in `old` that was deleted.
    Deleted(usize),
    /// usize is index in `new` that was inserted.
    Inserted(usize),
    /// usize is index in `new` that is unchanged.
    Unchanged(usize),
}

impl Edit {
    #[inline]
    pub const fn select<'a, T>(self, old: &'a [T], new: &'a [T]) -> &'a T {
        match self {
            Edit::Deleted(index) => &old[index],
            Edit::Inserted(index) => &new[index],
            Edit::Unchanged(index) => &new[index],
        }
    }

    #[inline]
    pub const fn select_mut<'a, T>(self, old: &'a mut [T], new: &'a mut [T]) -> &'a mut T {
        match self {
            Edit::Deleted(index) => &mut old[index],
            Edit::Inserted(index) => &mut new[index],
            Edit::Unchanged(index) => &mut new[index],
        }
    }

    #[inline]
    pub const fn edit_type(self) -> EditType {
        match self {
            Edit::Deleted(_) => EditType::Deleted,
            Edit::Inserted(_) => EditType::Inserted,
            Edit::Unchanged(_) => EditType::Unchanged,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EditRange {
    /// (start, end)
    Deleted(usize, usize),
    /// (start, end)
    Inserted(usize, usize),
    /// (start end)
    Unchanged(usize, usize),
}

impl EditRange {
    pub const fn start_from_edit(edit: Edit) -> Self {
        match edit {
            Edit::Deleted(index) => Self::Deleted(index, index + 1),
            Edit::Inserted(index) => Self::Inserted(index, index + 1),
            Edit::Unchanged(index) => Self::Unchanged(index, index + 1),
        }
    }

    pub const fn range(self) -> std::ops::Range<usize> {
        match self {
            EditRange::Deleted(start, end) => start..end,
            EditRange::Inserted(start, end) => start..end,
            EditRange::Unchanged(start, end) => start..end,
        }
    }

    pub fn select<'a, T>(self, old: &'a [T], new: &'a [T]) -> &'a [T] {
        match self {
            EditRange::Deleted(start, end) => &old[start..end],
            EditRange::Inserted(start, end) => &new[start..end],
            EditRange::Unchanged(start, end) => &new[start..end],
        }
    }

    pub fn select_mut<'a, T>(self, old: &'a mut [T], new: &'a mut [T]) -> &'a mut [T] {
        match self {
            EditRange::Deleted(start, end) => &mut old[start..end],
            EditRange::Inserted(start, end) => &mut new[start..end],
            EditRange::Unchanged(start, end) => &mut new[start..end],
        }
    }

    #[inline]
    pub const fn edit_type(self) -> EditType {
        match self {
            EditRange::Deleted(_, _) => EditType::Deleted,
            EditRange::Inserted(_, _) => EditType::Inserted,
            EditRange::Unchanged(_, _) => EditType::Unchanged,
        }
    }
}

struct RangeBuilder {
    current: Option<EditRange>,
    ranges: Vec<EditRange>,
}

impl RangeBuilder {
    /// Creates a new range builder with the initial state set to an [EditRange] start from the `edit`.
    /// 
    /// An [EditRange] start is a range where the end index is `1` more than the start.
    /// When you create an [EditRange] from an [Edit], you are creating a range that starts at the same index
    /// as the edit and also has the same variant as the edit. This works well since [EditRange] and [Edit] are dimorphic.
    const fn new_start_from_edit(edit: Edit) -> Self {
        Self {
            current: Some(EditRange::start_from_edit(edit)),
            ranges: Vec::new(),
        }
    }

    fn push(&mut self, edit: Edit) {
        self.current = Some(match self.current {
            Some(current) => {
                match (current, edit) {
                    (EditRange::Unchanged(start, end), Edit::Unchanged(index)) => {
                        debug_assert_eq!(end, index, "end does not match index");
                        EditRange::Unchanged(start, end + 1)
                    }
                    (EditRange::Inserted(start, end), Edit::Inserted(index)) => {
                        debug_assert_eq!(end, index, "end does not match index");
                        EditRange::Inserted(start, end + 1)
                    }
                    (EditRange::Deleted(start, end), Edit::Deleted(index)) => {
                        debug_assert_eq!(end, index, "end does not match index");
                        EditRange::Deleted(start, end + 1)
                    }
                    (range, edit) => {
                        self.ranges.push(range);
                        EditRange::start_from_edit(edit)
                    }
                }
            }
            None => {
                EditRange::start_from_edit(edit)
            }
        })
    }

    fn finalize(self) -> Vec<EditRange> {
        let Self {
            current,
            mut ranges,
        } = self;
        if let Some(current) = current {
            ranges.push(current);
        }
        ranges
    }
}

/// Builds an edit script in reverse order. This means that you would need to reverse the elements in order
/// to get the correct order. This is just a byproduct of the way the algorithm works, but it's handy
/// because it makes it simple to create an [EditRange] script.
fn diff_with_reverse_script<'a, T: PartialEq<T>>(old: &'a [T], new: &'a [T]) -> Vec<Edit> {
    let mut edits = Vec::with_capacity(old.len() + new.len());

    let mut table = EditTable::new(old, new);

    for x in 1..old.len() + 1 {
        table.set(x, 0, x);
    }
    for y in 1..new.len() + 1 {
        table.set(0, y, y);
    }

    for y in 1..new.len() + 1 {
        for x in 1..old.len() + 1 {
            if table.compare(x, y) {
                table.set(x, y, table.get(x - 1, y - 1));
            } else {
                let left = table.get(x - 1, y);
                let top = table.get(x, y - 1);
                table.set(x, y, left.min(top) + 1);
            }
        }
    }

    let mut xy = (old.len(), new.len());

    loop {
        match xy {
            (0, 0) => break,
            (0, mut y) => {
                while y != 0 {
                    edits.push(Edit::Inserted(y - 1));
                    y -= 1;
                }
                break;
            }
            (mut x, 0) => {
                while x != 0 {
                    edits.push(Edit::Deleted(x - 1));
                    x -= 1;
                }
                break;
            }
            (x, y) => {
                let curr = table.get(x, y);
                if curr == table.get(x, y - 1) + 1 {
                    edits.push(Edit::Inserted(y - 1));
                    xy.1 -= 1;
                } else if curr == table.get(x - 1, y) + 1 {
                    edits.push(Edit::Deleted(x - 1));
                    xy.0 -= 1;
                } else if table.compare(x, y)
                && curr == table.get(x - 1, y - 1) {
                    edits.push(Edit::Unchanged(y - 1));
                    xy = (x - 1, y - 1);
                } else {
                    unreachable!("Failed to select correct path. This should never happen.");
                }
            }
        }
    }

    edits
}

pub fn diff<T: PartialEq<T>>(old: &[T], new: &[T]) -> Vec<Edit> {
    let mut edits = diff_with_reverse_script(old, new);
    edits.reverse();
    edits
}

pub fn diff_ranges<T: PartialEq<T>>(old: &[T], new: &[T]) -> Vec<EditRange> {

    let mut edits = diff_with_reverse_script(old, new);

    let mut builder = if let Some(edit) = edits.pop() {
        RangeBuilder::new_start_from_edit(edit)
    } else {
        return Vec::new();
    };

    while let Some(edit) = edits.pop() {
        builder.push(edit);
    }

    builder.finalize()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn diff_test() {
        let seq_a = "ABC123ABC";
        let seq_b = "123ABC123";
        let edits = diff(seq_a.as_bytes(), seq_b.as_bytes());
        let edit_ranges = diff_ranges(seq_a.as_bytes(), seq_b.as_bytes());
        println!("Old: \"{seq_a}\"\nNew: \"{seq_b}\"");
        println!("--------------------------------");
        for range in edit_ranges {
            match range {
                EditRange::Deleted(start, end) => println!("-{}", &seq_a[start..end]),
                EditRange::Inserted(start, end) => println!("+{}", &seq_b[start..end]),
                EditRange::Unchanged(start, end) => println!("={}", &seq_b[start..end]),
            }
        }
        println!("--------------------------------");
        for edit in edits {
            match edit {
                Edit::Deleted(index) => println!("-{}", seq_a.as_bytes()[index] as char),
                Edit::Inserted(index) => println!("+{}", seq_b.as_bytes()[index] as char),
                Edit::Unchanged(index) => println!("={}", seq_b.as_bytes()[index] as char),
            }
        }
        println!("--------------------------------");
    }
}