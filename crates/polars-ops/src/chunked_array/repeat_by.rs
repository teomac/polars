use arrow::array::ListArray;
use polars_core::prelude::*;
use polars_core::with_match_physical_numeric_polars_type;

type LargeListArray = ListArray<i64>;

fn check_lengths(length_srs: usize, length_by: usize) -> PolarsResult<()> {
    polars_ensure!(
       (length_srs == length_by) | (length_by == 1) | (length_srs == 1),
       ComputeError: "repeat_by argument and the Series should have equal length, or at least one of them should have length 1. Series length {}, by length {}",
       length_srs, length_by
    );
    Ok(())
}

fn new_by(by: &IdxCa, len: usize) -> IdxCa {
    if let Some(x) = by.get(0) {
        let values = std::iter::repeat_n(x, len).collect::<Vec<IdxSize>>();
        IdxCa::new(PlSmallStr::EMPTY, values)
    } else {
        IdxCa::full_null(PlSmallStr::EMPTY, len)
    }
}

fn repeat_by_primitive<T>(ca: &ChunkedArray<T>, by: &IdxCa) -> PolarsResult<ListChunked>
where
    T: PolarsNumericType,
{
    check_lengths(ca.len(), by.len())?;

    match (ca.len(), by.len()) {
        (left_len, right_len) if left_len == right_len => {
            Ok(arity::binary(ca, by, |arr, by| {
                let iter = arr.into_iter().zip(by).map(|(opt_v, opt_by)| {
                    opt_by.map(|by| std::iter::repeat_n(opt_v.copied(), *by as usize))
                });

                // SAFETY: length of iter is trusted.
                unsafe {
                    LargeListArray::from_iter_primitive_trusted_len(
                        iter,
                        T::get_dtype().to_arrow(CompatLevel::newest()),
                    )
                }
            }))
        },
        (_, 1) => {
            let by = new_by(by, ca.len());
            repeat_by_primitive(ca, &by)
        },
        (1, _) => {
            let new_array = ca.new_from_index(0, by.len());
            repeat_by_primitive(&new_array, by)
        },
        // we have already checked the length
        _ => unreachable!(),
    }
}

fn repeat_by_bool(ca: &BooleanChunked, by: &IdxCa) -> PolarsResult<ListChunked> {
    check_lengths(ca.len(), by.len())?;

    match (ca.len(), by.len()) {
        (left_len, right_len) if left_len == right_len => {
            Ok(arity::binary(ca, by, |arr, by| {
                let iter = arr.into_iter().zip(by).map(|(opt_v, opt_by)| {
                    opt_by.map(|by| std::iter::repeat_n(opt_v, *by as usize))
                });

                // SAFETY: length of iter is trusted.
                unsafe { LargeListArray::from_iter_bool_trusted_len(iter) }
            }))
        },
        (_, 1) => {
            let by = new_by(by, ca.len());
            repeat_by_bool(ca, &by)
        },
        (1, _) => {
            let new_array = ca.new_from_index(0, by.len());
            repeat_by_bool(&new_array, by)
        },
        // we have already checked the length
        _ => unreachable!(),
    }
}

fn repeat_by_binary(ca: &BinaryChunked, by: &IdxCa) -> PolarsResult<ListChunked> {
    check_lengths(ca.len(), by.len())?;

    match (ca.len(), by.len()) {
        (left_len, right_len) if left_len == right_len => {
            Ok(arity::binary(ca, by, |arr, by| {
                let iter = arr.into_iter().zip(by).map(|(opt_v, opt_by)| {
                    opt_by.map(|by| std::iter::repeat_n(opt_v, *by as usize))
                });

                // SAFETY: length of iter is trusted.
                unsafe { LargeListArray::from_iter_binary_trusted_len(iter, ca.len()) }
            }))
        },
        (_, 1) => {
            let by = new_by(by, ca.len());
            repeat_by_binary(ca, &by)
        },
        (1, _) => {
            let new_array = ca.new_from_index(0, by.len());
            repeat_by_binary(&new_array, by)
        },
        // we have already checked the length
        _ => unreachable!(),
    }
}

pub fn repeat_by(s: &Series, by: &IdxCa) -> PolarsResult<ListChunked> {
    let s_phys = s.to_physical_repr();
    use DataType::*;
    let out = match s_phys.dtype() {
        Boolean => repeat_by_bool(s_phys.bool().unwrap(), by),
        String => {
            let ca = s_phys.str().unwrap();
            repeat_by_binary(&ca.as_binary(), by)
                .and_then(|ca| ca.apply_to_inner(&|s| unsafe { s.cast_unchecked(&String) }))
        },
        Binary => repeat_by_binary(s_phys.binary().unwrap(), by),
        dt if dt.is_primitive_numeric() => {
            with_match_physical_numeric_polars_type!(dt, |$T| {
                let ca: &ChunkedArray<$T> = s_phys.as_ref().as_ref().as_ref();
                repeat_by_primitive(ca, by)
            })
        },
        _ => polars_bail!(opq = repeat_by, s.dtype()),
    };
    out.and_then(|ca| {
        let logical_type = s.dtype();
        ca.apply_to_inner(&|s| unsafe { s.from_physical_unchecked(logical_type) })
    })
}
