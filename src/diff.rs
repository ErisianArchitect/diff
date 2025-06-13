use std::mem::MaybeUninit;

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EditType {
    Deleted = 0,
    Inserted = 1,
    Unchanged = 2,
}

#[repr(u8)]
#[derive(Debug)]
pub enum Edit<T> {
    Deleted(T) = 0,
    Inserted(T) = 1,
    Unchanged(T) = 2,
}

impl<T> Edit<T> {
    #[inline]
    pub fn map<R, F: FnOnce(T) -> R>(self, f: F) -> Edit<R> {
        match self {
            Edit::Deleted(value) => Edit::Deleted(f(value)),
            Edit::Inserted(value) => Edit::Inserted(f(value)),
            Edit::Unchanged(value) => Edit::Unchanged(f(value)),
        }
    }

    #[inline]
    pub const fn edit_type(&self) -> EditType {
        match self {
            Edit::Deleted(_) => EditType::Deleted,
            Edit::Inserted(_) => EditType::Inserted,
            Edit::Unchanged(_) => EditType::Unchanged,
        }
    }

    #[inline]
    pub fn is_type(&self, edit_type: EditType) -> bool {
        match (self, edit_type) {
            (Edit::Deleted(_), EditType::Deleted) => true,
            (Edit::Inserted(_), EditType::Inserted) => true,
            (Edit::Unchanged(_), EditType::Unchanged) => true,
            _ => false,
        }
    }
}

impl<T: Clone> Clone for Edit<T> {
    fn clone(&self) -> Self {
        match self {
            Edit::Deleted(value) => Edit::Deleted(value.clone()),
            Edit::Inserted(value) => Edit::Inserted(value.clone()),
            Edit::Unchanged(value) => Edit::Unchanged(value.clone()),
        }
    }
}

impl<T: Copy> Copy for Edit<T> {}

impl<T: PartialEq<T>> PartialEq<Self> for Edit<T> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Edit::Deleted(lhs), Edit::Deleted(rhs)) => lhs == rhs,
            (Edit::Inserted(lhs), Edit::Inserted(rhs)) => lhs == rhs,
            (Edit::Unchanged(lhs), Edit::Unchanged(rhs)) => lhs == rhs,
            _ => false,
        }
    }

    fn ne(&self, other: &Self) -> bool {
        match (self, other) {
            (Edit::Deleted(lhs), Edit::Deleted(rhs)) => lhs != rhs,
            (Edit::Inserted(lhs), Edit::Inserted(rhs)) => lhs != rhs,
            (Edit::Unchanged(lhs), Edit::Unchanged(rhs)) => lhs != rhs,
            _ => true,
        }
    }
}

impl<T: Eq> Eq for Edit<T> {}

impl<T: std::hash::Hash> std::hash::Hash for Edit<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            Edit::Deleted(value) => value.hash(state),
            Edit::Inserted(value) => value.hash(state),
            Edit::Unchanged(value) => value.hash(state),
        }
    }
}

pub type EditIndex = Edit<usize>;
pub type EditRange = Edit<(usize, usize)>;

impl EditIndex {
    #[inline]
    pub const fn select<'a, T>(self, old: &'a [T], new: &'a [T]) -> &'a T {
        match self {
            EditIndex::Deleted(index) => &old[index],
            EditIndex::Inserted(index) => &new[index],
            EditIndex::Unchanged(index) => &old[index],
        }
    }

    #[inline]
    pub const fn select_mut<'a, T>(self, old: &'a mut [T], new: &'a mut [T]) -> &'a mut T {
        match self {
            EditIndex::Deleted(index) => &mut old[index],
            EditIndex::Inserted(index) => &mut new[index],
            EditIndex::Unchanged(index) => &mut old[index],
        }
    }

    #[inline]
    pub const fn index(self) -> usize {
        match self {
            Edit::Deleted(index) => index,
            Edit::Inserted(index) => index,
            Edit::Unchanged(index) => index,
        }
    }

    /// Creates a range that starts at `index` and has a length of `1`.
    #[inline]
    pub fn range_start(self) -> EditRange {
        self.map(|index| (index, index + 1))
    }
}

impl EditRange {
    #[inline]
    pub const fn range(self) -> std::ops::Range<usize> {
        match self {
            EditRange::Deleted((start, end)) => start..end,
            EditRange::Inserted((start, end)) => start..end,
            EditRange::Unchanged((start, end)) => start..end,
        }
    }

    #[inline]
    pub fn select<'a, T>(self, old: &'a [T], new: &'a [T]) -> &'a [T] {
        match self {
            EditRange::Deleted((start, end)) => &old[start..end],
            EditRange::Inserted((start, end)) => &new[start..end],
            EditRange::Unchanged((start, end)) => &old[start..end],
        }
    }

    #[inline]
    pub fn select_mut<'a, T>(self, old: &'a mut [T], new: &'a mut [T]) -> &'a mut [T] {
        match self {
            EditRange::Deleted((start, end)) => &mut old[start..end],
            EditRange::Inserted((start, end)) => &mut new[start..end],
            EditRange::Unchanged((start, end)) => &mut old[start..end],
        }
    }
}

/// Stores only two rows at a time. Keeps track of the minimum
/// edit script length as the table is built.
/// This makes it possible to calculate the edit script length while building the
/// edit distance table without allocating an entire table for the edit lengths.
/// This utilizes the fact that the only possible cells that will be checked
/// are `(x, x)`, `(x - 1, y)`, `(x, y - 1)`, or `(x - 1, y - 1)`.
struct EditLenTracker {
    rows: [Vec<usize>; 2],
    /// The `index` determines which row is (virtually) `row[0]` and which is `row[1]`.
    /// If `index` is `0`, then `0 = 0` and `1 = 1`, if `index` is `1`, `0 = 1` and `1 = 0`.
    index: usize,
    y: usize,
}

impl EditLenTracker {
    fn new(width: usize) -> Self {
        Self {
            y: 0,
            index: 0,
            rows: [
                Vec::from_iter((0..width - 1).map(|_| 0)),
                Vec::from_iter((0..width - 1).map(|_| 0)),
            ]
        }
    }

    #[inline]
    fn next_row(&mut self) {
        self.index ^= 1;
        self.y += 1;
    }

    #[inline]
    fn index_of(&self, y: usize) -> usize {
        debug_assert!(y >= self.y && y < self.y + 2, "y={y} is out of range.");
        let index = y - self.y;
        // 0 ^ 0 = 0
        // 1 ^ 0 = 1
        // 0 ^ 1 = 1
        // 1 ^ 1 = 0
        self.index ^ index
    }

    fn get_len(&self, x: usize, y: usize) -> usize {
        match (x, y) {
            (0, 0) => 0,
            (0, y) => y,
            (x, 0) => x,
            (x, y) => {
                let row = self.index_of(y);
                self.rows[row][x - 1]
            }
        }
    }

    fn set_len(&mut self, x: usize, y: usize, value: usize) {
        match (x, y) {
            (0, 0) => (),
            (0, _y) => (),
            (_x, 0) => (),
            (x, y) => {
                let row = self.index_of(y);
                self.rows[row][x - 1] = value;
            }
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

    fn get_len(&self, x: usize, y: usize) -> usize {
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

    fn set_len(&mut self, x: usize, y: usize, value: usize) {
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
        let mut len_tracker = EditLenTracker::new(table.width);
        for y in 1..new.len() + 1 {
            for x in 1..old.len() + 1 {
                if table.compare(x, y) {
                    table.set_len(x, y, table.get_len(x - 1, y - 1));
                    len_tracker.set_len(x, y, len_tracker.get_len(x - 1, y - 1) + 1);
                } else {
                    let top = table.get_len(x, y - 1);
                    let left = table.get_len(x - 1, y);
                    let (dp, len) = if top <= left {
                        (top, len_tracker.get_len(x, y - 1))
                    } else {
                        (left, len_tracker.get_len(x - 1, y))
                    };
                    table.set_len(x, y, dp + 1);
                    len_tracker.set_len(x, y, len + 1);
                }
            }
            len_tracker.next_row();
        }
        (len_tracker.get_len(old.len(), new.len()), table)
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

    fn push(&mut self, edit: EditIndex) {
        self.current = Some(match (self.current, edit) {
            (Some(EditRange::Unchanged((start, end))), EditIndex::Unchanged(index)) => {
                debug_assert_eq!(end, index, "end does not match index");
                EditRange::Unchanged((start, end + 1))
            }
            (Some(EditRange::Inserted((start, end))), EditIndex::Inserted(index)) => {
                debug_assert_eq!(end, index, "end does not match index");
                EditRange::Inserted((start, end + 1))
            }
            (Some(EditRange::Deleted((start, end))), EditIndex::Deleted(index)) => {
                debug_assert_eq!(end, index, "end does not match index");
                EditRange::Deleted((start, end + 1))
            }
            (None, edit) => {
                edit.range_start()
            }
            (Some(range), edit) => {
                self.ranges.push(range);
                edit.range_start()
            }
        });
    }

    fn finish(self) -> Vec<EditRange> {
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

    #[inline]
    fn push(&mut self, edit_type: EditType) {
        if self.current_type != Some(edit_type) {
            self.count += 1;
            self.current_type = Some(edit_type);
        }
    }
}

pub struct EditScript {
    range_count: usize,
    edits: Vec<EditIndex>,
}

impl EditScript {
    #[inline]
    const fn new(range_count: usize, edits: Vec<EditIndex>) -> Self {
        Self {
            range_count,
            edits,
        }
    }

    #[inline]
    pub fn edits(&self) -> &[EditIndex] {
        self.edits.as_slice()
    }

    /// The range count is the minimum number of [EditRange]s to create this
    /// script.
    #[inline]
    pub fn range_count(&self) -> usize {
        self.range_count
    }

    #[inline]
    pub fn take_edits(self) -> Vec<EditIndex> {
        self.edits
    }

    /// Builds a new [EditRange] script.
    pub fn build_range_script(&self) -> Vec<EditRange> {
        let mut builder = DiffRangesBuilder::new(self.range_count);
        for edit in self.edits.iter().cloned() {
            builder.push(edit);
        }
        builder.finish()
    }

    pub fn iter(&self) -> std::iter::Cloned<std::slice::Iter<'_, EditIndex>> {
        self.edits.iter().cloned()
    }
}

impl IntoIterator for EditScript {
    type Item = EditIndex;
    type IntoIter = <Vec::<EditIndex> as IntoIterator>::IntoIter;
    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.edits.into_iter()
    }
}

/// Edit scripts are built in reverse order in the algorithm, and the number of edits is
/// known beforehand, so we can utilize these properties to create a backwards filling
/// script builder.
struct EditScriptBuilder {
    script: Box<[MaybeUninit<EditIndex>]>,
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

    fn push(&mut self, edit: EditIndex) {
        debug_assert_ne!(self.index, 0, "Overflow.");
        let index = self.index - 1;
        self.index = index;
        self.range_counter.push(edit.edit_type());
        self.script[index] = MaybeUninit::new(edit);
    }

    fn finish(self) -> EditScript {
        debug_assert_eq!(self.index, 0, "Buffer was not fully initialized before calling .finish()");
        EditScript {
            range_count: self.range_counter.count,
            edits: unsafe {
                // So long as the entire script has been initialized, this should be safe.
                // The EditScript construct should only be used in places where this can be guaranteed.
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
            return EditScript::new(1, (0..len).map(|i| EditIndex::Inserted(i)).collect());
        }
        (len, 0) => {
            return EditScript::new(1, (0..len).map(|i| EditIndex::Deleted(i)).collect());
        }
        (old_len, new_len) if old_len == new_len && old == new => {
            return EditScript::new(1, (0..old_len).map(|i| EditIndex::Unchanged(i)).collect());
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
                    script_builder.push(EditIndex::Inserted(y - 1));
                    y -= 1;
                }
                break;
            }
            (mut x, 0) => {
                while x != 0 {
                    script_builder.push(EditIndex::Deleted(x - 1));
                    x -= 1;
                }
                break;
            }
            (x, y) => {
                let curr = table.get_len(x, y);
                if table.compare(x, y)
                && curr == table.get_len(x - 1, y - 1) {
                    script_builder.push(EditIndex::Unchanged(x - 1));
                    xy = (x - 1, y - 1);
                } else if curr == table.get_len(x, y - 1) + 1 {
                    script_builder.push(EditIndex::Inserted(y - 1));
                    xy.1 -= 1;
                } else if curr == table.get_len(x - 1, y) + 1 {
                    script_builder.push(EditIndex::Deleted(x - 1));
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
        (0, len) => return vec![EditRange::Inserted((0, len))],
        (len, 0) => return vec![EditRange::Deleted((0, len))],
        (old_len, new_len) if old_len == new_len && old == new => {
            return vec![EditRange::Unchanged((0, old_len))];
        }
        _ => (),                                 
    }
    diff(old, new).build_range_script()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn edit_value_map() {
        let value = Edit::Inserted(0u32);
        let value = value.map(|value| {
            value..value + 1
        });
        assert_eq!(value, Edit::Inserted(0..1));
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
            "1",
            "Foo",
            "Bar",
            "New edition",
            "Baz",
            "Test",
        ];
        let seq_b: &[&str] = &[
            "hello world",
            "The quick brown fox jumps over the lazy dog.",
            "1",
            "2",
            "3",
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
        let ranges_script = edit_script.build_range_script();
        assert_eq!(ranges_script.len(), ranges_script.capacity());
        assert_eq!(edit_ranges, ranges_script);
        let edits = edit_script.take_edits();
        println!("--------------------------------");
        for edit in edits {
            match edit {
                EditIndex::Deleted(index) => println!("-{}", seq_a[index]),
                EditIndex::Inserted(index) => println!("+{}", seq_b[index]),
                EditIndex::Unchanged(index) => println!("={}", seq_a[index]),
            }
        }
        println!("--------------------------------");
        for range in edit_ranges {
            let (prefix, slice) = match range {
                EditRange::Deleted((start, end)) => {
                    (
                        '-',
                        &seq_a[start..end],
                    )
                },
                EditRange::Inserted((start, end)) => {
                    (
                        '+',
                        &seq_b[start..end],
                    )
                },
                EditRange::Unchanged((start, end)) => {
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