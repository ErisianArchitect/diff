use std::mem::MaybeUninit;


/// Stores only two rows at a time. Keeps track of the minimum
/// edit script length as the table is built.
/// This makes it possible to calculate the edit script length while building the
/// edit distance table without allocating an entire table for the edit lengths.
/// This utilizes the fact that the only possible cells that will be checked
/// are `(x, x)`, `(x - 1, y)`, `(x, y - 1)`, or `(x - 1, y - 1)`.
struct EditLenTracker {
    rows: [Vec<usize>; 2],
    /// The indices into `rows`.
    indices: [usize; 2],
    y: usize,
    /// After the table is built, this value should store the last length (which should be the bottom-right).
    /// If this value represents the value at the bottom-right, that means that this is the final edit script's length.
    last_len: usize,
}

impl EditLenTracker {
    fn new(width: usize, y: usize) -> Self {
        Self {
            y,
            last_len: 0,
            indices: [0, 1],
            rows: [
                Vec::from_iter((0..width).map(|_| 0)),
                Vec::from_iter((0..width).map(|_| 0)),
            ]
        }
    }

    fn next_row(&mut self) {
        // Since indices is either [0, 1] or [1, 0],
        // we can XOR both of them by 1, which has the same effect as swapping them.
        self.indices[0] ^= 1;
        self.indices[1] ^= 1;
        self.y += 1;
    }

    fn get(&self, x: usize, y: usize) -> usize {
        match (x, y) {
            (0, 0) => 0,
            (0, y) => y,
            (x, 0) => x,
            (x, y) => {
                let Some(index_index) = y.checked_sub(self.y) else {
                    // The reason this should be unreachable is because the EditLenTracker
                    // is not public, and I intend to not use it in an erroneous way.
                    // This will tell me if I've used it wrong.
                    unreachable!("y={y} is out of range. This should be unreachable.");
                };
                debug_assert!(index_index < 2, "y={y} is out of range.");
                let index = self.indices[index_index];
                self.rows[index][x]
            }
        }
    }

    fn set(&mut self, x: usize, y: usize, value: usize) {
        match (x, y) {
            (0, 0) => (),
            (0, _y) => (),
            (_x, 0) => (),
            (x, y) => {
                let Some(index_index) = y.checked_sub(self.y) else {
                    panic!("y={y} is out of range.");
                };
                debug_assert!(index_index < 2, "y={y} is out of range.");
                let index = self.indices[index_index];
                self.last_len = value;
                self.rows[index][x] = value;
            }
        }
    }
}

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EditType {
    Deleted = 0,
    Inserted = 1,
    Unchanged = 2,
}

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Edit {
    /// usize is index in `old` that was deleted.
    Deleted(usize) = 0,
    /// usize is index in `new` that was inserted.
    Inserted(usize) = 1,
    /// usize is index in `old` that is unchanged.
    Unchanged(usize) = 2,
}

impl Edit {
    #[inline]
    pub const fn select<'a, T>(self, old: &'a [T], new: &'a [T]) -> &'a T {
        match self {
            Edit::Deleted(index) => &old[index],
            Edit::Inserted(index) => &new[index],
            Edit::Unchanged(index) => &old[index],
        }
    }

    #[inline]
    pub const fn select_mut<'a, T>(self, old: &'a mut [T], new: &'a mut [T]) -> &'a mut T {
        match self {
            Edit::Deleted(index) => &mut old[index],
            Edit::Inserted(index) => &mut new[index],
            Edit::Unchanged(index) => &mut old[index],
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

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EditRange {
    /// (start, end)
    Deleted(usize, usize) = 0,
    /// (start, end)
    Inserted(usize, usize) = 1,
    /// (start end)
    Unchanged(usize, usize) = 2,
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
            EditRange::Unchanged(start, end) => &old[start..end],
        }
    }

    pub fn select_mut<'a, T>(self, old: &'a mut [T], new: &'a mut [T]) -> &'a mut [T] {
        match self {
            EditRange::Deleted(start, end) => &mut old[start..end],
            EditRange::Inserted(start, end) => &mut new[start..end],
            EditRange::Unchanged(start, end) => &mut old[start..end],
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

#[derive(Debug)]
pub enum EditValue<T> {
    Deleted(T),
    Inserted(T),
    Unchanged(T),
}

impl<T> EditValue<T> {
    pub fn map<R, F: FnOnce(T) -> R>(self, f: F) -> EditValue<R> {
        match self {
            EditValue::Deleted(value) => EditValue::Deleted(f(value)),
            EditValue::Inserted(value) => EditValue::Inserted(f(value)),
            EditValue::Unchanged(value) => EditValue::Unchanged(f(value)),
        }
    }

    pub fn edit_type(&self) -> EditType {
        match self {
            EditValue::Deleted(_) => EditType::Deleted,
            EditValue::Inserted(_) => EditType::Inserted,
            EditValue::Unchanged(_) => EditType::Unchanged,
        }
    }
}

impl<T: Clone> Clone for EditValue<T> {
    fn clone(&self) -> Self {
        match self {
            EditValue::Deleted(value) => EditValue::Deleted(value.clone()),
            EditValue::Inserted(value) => EditValue::Inserted(value.clone()),
            EditValue::Unchanged(value) => EditValue::Unchanged(value.clone()),
        }
    }
}

impl<T: Copy> Copy for EditValue<T> {}

impl<T: PartialEq<T>> PartialEq<Self> for EditValue<T> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (EditValue::Deleted(lhs), EditValue::Deleted(rhs)) => lhs == rhs,
            (EditValue::Inserted(lhs), EditValue::Inserted(rhs)) => lhs == rhs,
            (EditValue::Unchanged(lhs), EditValue::Unchanged(rhs)) => lhs == rhs,
            _ => false,
        }
    }

    fn ne(&self, other: &Self) -> bool {
        match (self, other) {
            (EditValue::Deleted(lhs), EditValue::Deleted(rhs)) => lhs != rhs,
            (EditValue::Inserted(lhs), EditValue::Inserted(rhs)) => lhs != rhs,
            (EditValue::Unchanged(lhs), EditValue::Unchanged(rhs)) => lhs != rhs,
            _ => true,
        }
    }
}

impl<T: Eq> Eq for EditValue<T> {}

impl<T: std::hash::Hash> std::hash::Hash for EditValue<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            EditValue::Deleted(value) => value.hash(state),
            EditValue::Inserted(value) => value.hash(state),
            EditValue::Unchanged(value) => value.hash(state),
        }
    }
}

/// [EditTable] is used as a lookup table to see how many edits are required to get
/// to any given point in the table.
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
        let cell_count = old.len() * new.len();
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
            // I've thought about this, and it's a little confusing. I do believe that (0, _) and (_, 0) should return
            // false since you could consider them to be matching the null element with a non-null element, which should
            // not be equal.
            (0, _) => false,
            (_, 0) => false,
            (x, y) => {
                self.old[x - 1] == self.new[y - 1]
            }
        }
    }

    fn index_at(&self, x: usize, y: usize) -> usize {
        debug_assert!(x < self.width && y < self.height);
        ((y - 1) * (self.width - 1)) + (x - 1)
    }

    fn get(&self, x: usize, y: usize) -> usize {
        match (x, y) {
            (0, 0) => 0,
            (0, y) => y,
            (x, 0) => x,
            (x, y) => {
                let index = self.index_at(x, y);
                self.cells[index]
            }
        }
    }

    fn set(&mut self, x: usize, y: usize, value: usize) {
        match (x, y) {
            // If you get this `unreachable` panic, that means that you tried to
            // set a value in a row/column that doesn't exist. You can't set values at x==0 || y == 0.
            (0, _) => unreachable!("This should not happen. If this happens, change your code."),
            (_, 0) => unreachable!("This should not happen. If this happens, change your code."),
            (x, y) => {
                let index = self.index_at(x, y);
                self.cells[index] = value;
            }
        }
    }

    /// Returns (edit_script_length, EditTable).
    fn build(old: &'a [T], new: &'a [T]) -> (usize, Self) {
        let mut table = EditTable::new(old, new);
        let mut len_tracker = EditLenTracker::new(table.width, 0);
        for y in 1..new.len() + 1 {
            for x in 1..old.len() + 1 {
                if table.compare(x, y) {
                    table.set(x, y, table.get(x - 1, y - 1));
                    len_tracker.set(x, y, len_tracker.get(x - 1, y - 1) + 1);
                } else {
                    let top = table.get(x, y - 1);
                    let left = table.get(x - 1, y);
                    let (dp, len) = if top <= left {
                        (top, len_tracker.get(x, y - 1))
                    } else {
                        (left, len_tracker.get(x - 1, y))
                    };
                    table.set(x, y, dp + 1);
                    len_tracker.set(x, y, len + 1);
                }
            }
            len_tracker.next_row();
        }
        (len_tracker.last_len, table)
    }
}

struct DiffRangesBuilder {
    current: Option<EditRange>,
    ranges: Vec<EditRange>,
}

impl DiffRangesBuilder {
    fn new(capacity: usize) -> Self {
        Self {
            current: None,
            ranges: Vec::with_capacity(capacity),
        }
    }

    fn push(&mut self, edit: Edit) {
        self.current = Some(match (self.current, edit) {
            (Some(EditRange::Unchanged(start, end)), Edit::Unchanged(index)) => {
                debug_assert_eq!(end, index, "end does not match index");
                EditRange::Unchanged(start, end + 1)
            }
            (Some(EditRange::Inserted(start, end)), Edit::Inserted(index)) => {
                debug_assert_eq!(end, index, "end does not match index");
                EditRange::Inserted(start, end + 1)
            }
            (Some(EditRange::Deleted(start, end)), Edit::Deleted(index)) => {
                debug_assert_eq!(end, index, "end does not match index");
                EditRange::Deleted(start, end + 1)
            }
            (Some(range), edit) => {
                self.ranges.push(range);
                EditRange::start_from_edit(edit)
            }
            (None, edit) => {
                EditRange::start_from_edit(edit)
            }
        });
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

struct EditRangeCounter {
    current_type: Option<EditType>,
    count: usize,
}

impl EditRangeCounter {
    fn new() -> Self {
        Self {
            current_type: None,
            count: 0,
        }
    }

    /// Sets the current type and increases the count by `1`.
    #[inline]
    fn set_current_type(&mut self, edit_type: EditType) {
        self.count += 1;
        self.current_type = Some(edit_type);
    }

    /// Works as a pass through. It determines whether to increase the count
    /// based on the input value, then returns the input value untouched.
    #[inline]
    fn record(&mut self, edit_type: EditType) {
        if let Some(current_type) = self.current_type {
            if current_type != edit_type {
                self.set_current_type(edit_type);
            }
        } else {
            self.set_current_type(edit_type);
        }
    }
}

pub struct EditScript {
    range_count: usize,
    edits: Vec<Edit>,
}

impl EditScript {
    #[inline]
    const fn new(range_count: usize, edits: Vec<Edit>) -> Self {
        Self {
            range_count,
            edits,
        }
    }

    #[inline]
    pub fn edits(&self) -> &[Edit] {
        self.edits.as_slice()
    }

    /// The range count is the minimum number of [EditRange]s to create this
    /// script.
    #[inline]
    pub fn range_count(&self) -> usize {
        self.range_count
    }

    #[inline]
    pub fn take_edits(self) -> Vec<Edit> {
        self.edits
    }

    pub fn range_script(&self) -> Vec<EditRange> {
        let mut builder = DiffRangesBuilder::new(self.range_count);
        for edit in self.edits.iter().cloned() {
            builder.push(edit);
        }
        builder.finalize()
    }
}

/// Edit scripts are built in reverse order in the algorithm, and the number of edits is
/// known beforehand, so we can utilize these properties to create a backwards filling
/// script builder.
struct EditScriptBuilder {
    script: Box<[MaybeUninit<Edit>]>,
    index: usize,
    range_counter: EditRangeCounter,
}

impl EditScriptBuilder {
    fn new(capacity: usize) -> Self {
        Self {
            script: Box::new_uninit_slice(capacity),
            index: capacity,
            range_counter: EditRangeCounter::new(),
        }
    }

    fn push(&mut self, edit: Edit) {
        debug_assert_ne!(self.index, 0, "Overflow.");
        let index = self.index - 1;
        self.index = index;
        self.range_counter.record(edit.edit_type());
        self.script[index] = MaybeUninit::new(edit);
    }

    fn finish(self) -> EditScript {
        debug_assert_eq!(self.index, 0);
        EditScript {
            range_count: self.range_counter.count,
            edits: unsafe {
                Vec::from(self.script.assume_init())
            },
        }
    }
}

pub fn diff<'a, T: PartialEq<T>>(old: &'a [T], new: &'a [T]) -> EditScript {
    match (old.len(), new.len()) {
        (0, 0) => {
            return EditScript::new(0, Vec::new());
        }
        (0, len) => {
            return EditScript::new(1, (0..len).map(|i| Edit::Inserted(i)).collect());
        }
        (len, 0) => {
            return EditScript::new(1, (0..len).map(|i| Edit::Deleted(i)).collect());
        }
        (old_len, new_len) if old_len == new_len && old == new => {
            return EditScript::new(1, (0..old_len).map(|i| Edit::Unchanged(i)).collect());
        }
        _ => (),
    }
    let (script_len, table) = EditTable::build(old, new);
    let mut script_builder = EditScriptBuilder::new(script_len);
    let mut xy = (old.len(), new.len());
    loop {
        match xy {
            (0, 0) => break,
            (0, mut y) => {
                while y != 0 {
                    script_builder.push(Edit::Inserted(y - 1));
                    y -= 1;
                }
                break;
            }
            (mut x, 0) => {
                while x != 0 {
                    script_builder.push(Edit::Deleted(x - 1));
                    x -= 1;
                }
                break;
            }
            (x, y) => {
                let curr = table.get(x, y);
                if table.compare(x, y)
                && curr == table.get(x - 1, y - 1) {
                    script_builder.push(Edit::Unchanged(x - 1));
                    xy = (x - 1, y - 1);
                } else if curr == table.get(x, y - 1) + 1 {
                    script_builder.push(Edit::Inserted(y - 1));
                    xy.1 -= 1;
                } else if curr == table.get(x - 1, y) + 1 {
                    script_builder.push(Edit::Deleted(x - 1));
                    xy.0 -= 1;
                } else {
                    unreachable!("Failed to select correct path. This should never happen.");
                }
            }
        }
    }
    script_builder.finish()
}

pub fn diff_ranges<T: PartialEq<T>>(old: &[T], new: &[T]) -> Vec<EditRange> {
    match (old.len(), new.len()) {
        (0, 0) => return vec![],
        (0, len) => return vec![EditRange::Inserted(0, len)],
        (len, 0) => return vec![EditRange::Deleted(0, len)],
        (old_len, new_len) if old_len == new_len && old == new => {
            return vec![EditRange::Unchanged(0, old_len)];
        }
        _ => (),
    }
    let edit_script = diff(old, new);
    edit_script.range_script()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn edit_value_map() {
        let value = EditValue::Inserted(0u32);
        let value = value.map(|value| {
            value..value + 1
        });
        assert_eq!(value, EditValue::Inserted(0..1));
    }

    #[test]
    fn swap_indices_test() {
        let mut indices: [usize; 2] = [0, 1];
        
        fn swap(indices: &mut [usize; 2]) {
            indices[0] ^= 1;
            indices[1] ^= 1;
        }

        assert_eq!(indices, [0, 1]);
        swap(&mut indices);
        assert_eq!(indices, [1, 0]);
        swap(&mut indices);
        assert_eq!(indices, [0, 1]);
        swap(&mut indices);
        assert_eq!(indices, [1, 0]);
        swap(&mut indices);
        assert_eq!(indices, [0, 1]);

    }

    #[test]
    fn diff_test() {
        let seq_a: &[&str] = &[
            "Hello, World!",
            "The quick brown fox jumps over the lazy dog.",
            "Foo",
            "Bar",
            // "New edition",
            "Baz",
            "Test",
        ];
        let seq_b: &[&str] = &[
            "hello world",
            "The quick brown fox jumps over the lazy dog.",
            "Bar",
            "Baz",
            "Foo",
            "Bar",
            "Baz",
        ];
        let edit_script = diff(seq_a, seq_b);
        assert_eq!(edit_script.edits.len(), edit_script.edits.capacity());
        let edit_ranges = diff_ranges(seq_a, seq_b);
        assert_eq!(edit_ranges.len(), edit_ranges.capacity());
        let ranges_script = edit_script.range_script();
        assert_eq!(ranges_script.len(), ranges_script.capacity());
        assert_eq!(edit_ranges, ranges_script);
        let edits = edit_script.take_edits();
        println!("--------------------------------");
        for edit in edits {
            match edit {
                Edit::Deleted(index) => println!("-{}", seq_a[index]),
                Edit::Inserted(index) => println!("+{}", seq_b[index]),
                Edit::Unchanged(index) => println!("={}", seq_a[index]),
            }
        }
        println!("--------------------------------");
        for range in edit_ranges {
            let (prefix, slice) = match range {
                EditRange::Deleted(start, end) => {
                    (
                        '-',
                        &seq_a[start..end],
                    )
                },
                EditRange::Inserted(start, end) => {
                    (
                        '+',
                        &seq_b[start..end],
                    )
                },
                EditRange::Unchanged(start, end) => {
                    (
                        '=',
                        &seq_a[start..end],
                    )
                },
            };
            for &line in slice {
                println!("{}│ {}", prefix, line);
            }
        }
        println!("--------------------------------");
    }
}