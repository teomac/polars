use std::fmt::Formatter;
use std::ops::Deref;
use std::sync::Arc;

#[cfg(feature = "python")]
use polars_utils::pl_serialize::deserialize_map_bytes;
#[cfg(feature = "serde")]
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use super::*;

/// A wrapper trait for any closure `Fn(Vec<Series>) -> PolarsResult<Series>`
pub trait ColumnsUdf: Send + Sync {
    fn as_any(&self) -> &dyn std::any::Any {
        unimplemented!("as_any not implemented for this 'opaque' function")
    }

    fn call_udf(&self, s: &mut [Column]) -> PolarsResult<Option<Column>>;

    fn try_serialize(&self, _buf: &mut Vec<u8>) -> PolarsResult<()> {
        polars_bail!(ComputeError: "serialization not supported for this 'opaque' function")
    }
}

#[cfg(feature = "serde")]
impl Serialize for SpecialEq<Arc<dyn ColumnsUdf>> {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        use serde::ser::Error;
        let mut buf = vec![];
        self.0
            .try_serialize(&mut buf)
            .map_err(|e| S::Error::custom(format!("{e}")))?;
        serializer.serialize_bytes(&buf)
    }
}

#[cfg(feature = "serde")]
impl<T: Serialize + Clone> Serialize for LazySerde<T> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            Self::Deserialized(t) => t.serialize(serializer),
            Self::Bytes(b) => b.serialize(serializer),
        }
    }
}

#[cfg(feature = "serde")]
impl<'a, T: Deserialize<'a> + Clone> Deserialize<'a> for LazySerde<T> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'a>,
    {
        let buf = bytes::Bytes::deserialize(deserializer)?;
        Ok(Self::Bytes(buf))
    }
}

#[cfg(feature = "serde")]
// impl<T: Deserialize> Deserialize for crate::dsl::expr::LazySerde<T> {
impl<'a> Deserialize<'a> for SpecialEq<Arc<dyn ColumnsUdf>> {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'a>,
    {
        use serde::de::Error;
        #[cfg(feature = "python")]
        {
            deserialize_map_bytes(deserializer, |buf| {
                if buf.starts_with(crate::dsl::python_dsl::PYTHON_SERDE_MAGIC_BYTE_MARK) {
                    let udf = crate::dsl::python_dsl::PythonUdfExpression::try_deserialize(&buf)
                        .map_err(|e| D::Error::custom(format!("{e}")))?;
                    Ok(SpecialEq::new(udf))
                } else {
                    Err(D::Error::custom(
                        "deserialization not supported for this 'opaque' function",
                    ))
                }
            })?
        }
        #[cfg(not(feature = "python"))]
        {
            _ = deserializer;

            Err(D::Error::custom(
                "deserialization not supported for this 'opaque' function",
            ))
        }
    }
}

impl<F> ColumnsUdf for F
where
    F: Fn(&mut [Column]) -> PolarsResult<Option<Column>> + Send + Sync,
{
    fn call_udf(&self, s: &mut [Column]) -> PolarsResult<Option<Column>> {
        self(s)
    }
}

impl Debug for dyn ColumnsUdf {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "ColumnUdf")
    }
}

/// A wrapper trait for any binary closure `Fn(Column, Column) -> PolarsResult<Column>`
pub trait ColumnBinaryUdf: Send + Sync {
    fn call_udf(&self, a: Column, b: Column) -> PolarsResult<Column>;
}

impl<F> ColumnBinaryUdf for F
where
    F: Fn(Column, Column) -> PolarsResult<Column> + Send + Sync,
{
    fn call_udf(&self, a: Column, b: Column) -> PolarsResult<Column> {
        self(a, b)
    }
}

impl Debug for dyn ColumnBinaryUdf {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "ColumnBinaryUdf")
    }
}

impl Default for SpecialEq<Arc<dyn ColumnBinaryUdf>> {
    fn default() -> Self {
        panic!("implementation error");
    }
}

impl Default for SpecialEq<Arc<dyn BinaryUdfOutputField>> {
    fn default() -> Self {
        let output_field = move |_: &Schema, _: Context, _: &Field, _: &Field| None;
        SpecialEq::new(Arc::new(output_field))
    }
}

pub trait RenameAliasFn: Send + Sync {
    fn call(&self, name: &PlSmallStr) -> PolarsResult<PlSmallStr>;

    fn try_serialize(&self, _buf: &mut Vec<u8>) -> PolarsResult<()> {
        polars_bail!(ComputeError: "serialization not supported for this renaming function")
    }
}

impl<F> RenameAliasFn for F
where
    F: Fn(&PlSmallStr) -> PolarsResult<PlSmallStr> + Send + Sync,
{
    fn call(&self, name: &PlSmallStr) -> PolarsResult<PlSmallStr> {
        self(name)
    }
}

impl Debug for dyn RenameAliasFn {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "RenameAliasFn")
    }
}

#[derive(Clone)]
/// Wrapper type that has special equality properties
/// depending on the inner type specialization
pub struct SpecialEq<T>(T);

#[cfg(feature = "serde")]
impl Serialize for SpecialEq<Series> {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.0.serialize(serializer)
    }
}

#[cfg(feature = "serde")]
impl<'a> Deserialize<'a> for SpecialEq<Series> {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'a>,
    {
        let t = Series::deserialize(deserializer)?;
        Ok(SpecialEq(t))
    }
}

#[cfg(feature = "serde")]
impl<T: Serialize> Serialize for SpecialEq<Arc<T>> {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.0.serialize(serializer)
    }
}

#[cfg(feature = "serde")]
impl<'a, T: Deserialize<'a>> Deserialize<'a> for SpecialEq<Arc<T>> {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'a>,
    {
        let t = T::deserialize(deserializer)?;
        Ok(SpecialEq(Arc::new(t)))
    }
}

impl<T> SpecialEq<T> {
    pub fn new(val: T) -> Self {
        SpecialEq(val)
    }
}

impl<T: ?Sized> PartialEq for SpecialEq<Arc<T>> {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl<T> Eq for SpecialEq<Arc<T>> {}

impl PartialEq for SpecialEq<Series> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<T> Debug for SpecialEq<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "no_eq")
    }
}

impl<T> Deref for SpecialEq<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

pub trait BinaryUdfOutputField: Send + Sync {
    fn get_field(
        &self,
        input_schema: &Schema,
        cntxt: Context,
        field_a: &Field,
        field_b: &Field,
    ) -> Option<Field>;
}

impl<F> BinaryUdfOutputField for F
where
    F: Fn(&Schema, Context, &Field, &Field) -> Option<Field> + Send + Sync,
{
    fn get_field(
        &self,
        input_schema: &Schema,
        cntxt: Context,
        field_a: &Field,
        field_b: &Field,
    ) -> Option<Field> {
        self(input_schema, cntxt, field_a, field_b)
    }
}

pub trait FunctionOutputField: Send + Sync {
    fn get_field(
        &self,
        input_schema: &Schema,
        cntxt: Context,
        fields: &[Field],
    ) -> PolarsResult<Field>;

    fn try_serialize(&self, _buf: &mut Vec<u8>) -> PolarsResult<()> {
        polars_bail!(ComputeError: "serialization not supported for this output field")
    }
}

pub type GetOutput = SpecialEq<Arc<dyn FunctionOutputField>>;

impl Default for GetOutput {
    fn default() -> Self {
        SpecialEq::new(Arc::new(
            |_input_schema: &Schema, _cntxt: Context, fields: &[Field]| Ok(fields[0].clone()),
        ))
    }
}

impl GetOutput {
    pub fn same_type() -> Self {
        Default::default()
    }

    pub fn first() -> Self {
        SpecialEq::new(Arc::new(
            |_input_schema: &Schema, _cntxt: Context, fields: &[Field]| Ok(fields[0].clone()),
        ))
    }

    pub fn from_type(dt: DataType) -> Self {
        SpecialEq::new(Arc::new(move |_: &Schema, _: Context, flds: &[Field]| {
            Ok(Field::new(flds[0].name().clone(), dt.clone()))
        }))
    }

    pub fn map_field<F: 'static + Fn(&Field) -> PolarsResult<Field> + Send + Sync>(f: F) -> Self {
        SpecialEq::new(Arc::new(move |_: &Schema, _: Context, flds: &[Field]| {
            f(&flds[0])
        }))
    }

    pub fn map_fields<F: 'static + Fn(&[Field]) -> PolarsResult<Field> + Send + Sync>(
        f: F,
    ) -> Self {
        SpecialEq::new(Arc::new(move |_: &Schema, _: Context, flds: &[Field]| {
            f(flds)
        }))
    }

    pub fn map_dtype<F: 'static + Fn(&DataType) -> PolarsResult<DataType> + Send + Sync>(
        f: F,
    ) -> Self {
        SpecialEq::new(Arc::new(move |_: &Schema, _: Context, flds: &[Field]| {
            let mut fld = flds[0].clone();
            let new_type = f(fld.dtype())?;
            fld.coerce(new_type);
            Ok(fld)
        }))
    }

    pub fn float_type() -> Self {
        Self::map_dtype(|dt| {
            Ok(match dt {
                DataType::Float32 => DataType::Float32,
                _ => DataType::Float64,
            })
        })
    }

    pub fn super_type() -> Self {
        Self::map_dtypes(|dtypes| {
            let mut st = dtypes[0].clone();
            for dt in &dtypes[1..] {
                st = try_get_supertype(&st, dt)?;
            }
            Ok(st)
        })
    }

    pub fn map_dtypes<F>(f: F) -> Self
    where
        F: 'static + Fn(&[&DataType]) -> PolarsResult<DataType> + Send + Sync,
    {
        SpecialEq::new(Arc::new(move |_: &Schema, _: Context, flds: &[Field]| {
            let mut fld = flds[0].clone();
            let dtypes = flds.iter().map(|fld| fld.dtype()).collect::<Vec<_>>();
            let new_type = f(&dtypes)?;
            fld.coerce(new_type);
            Ok(fld)
        }))
    }
}

impl<F> FunctionOutputField for F
where
    F: Fn(&Schema, Context, &[Field]) -> PolarsResult<Field> + Send + Sync,
{
    fn get_field(
        &self,
        input_schema: &Schema,
        cntxt: Context,
        fields: &[Field],
    ) -> PolarsResult<Field> {
        self(input_schema, cntxt, fields)
    }
}

#[cfg(feature = "serde")]
impl Serialize for GetOutput {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        use serde::ser::Error;
        let mut buf = vec![];
        self.0
            .try_serialize(&mut buf)
            .map_err(|e| S::Error::custom(format!("{e}")))?;
        serializer.serialize_bytes(&buf)
    }
}

#[cfg(feature = "serde")]
impl<'a> Deserialize<'a> for GetOutput {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'a>,
    {
        use serde::de::Error;
        #[cfg(feature = "python")]
        {
            deserialize_map_bytes(deserializer, |buf| {
                if buf.starts_with(self::python_dsl::PYTHON_SERDE_MAGIC_BYTE_MARK) {
                    let get_output = self::python_dsl::PythonGetOutput::try_deserialize(&buf)
                        .map_err(|e| D::Error::custom(format!("{e}")))?;
                    Ok(SpecialEq::new(get_output))
                } else {
                    Err(D::Error::custom(
                        "deserialization not supported for this output field",
                    ))
                }
            })?
        }
        #[cfg(not(feature = "python"))]
        {
            _ = deserializer;

            Err(D::Error::custom(
                "deserialization not supported for this output field",
            ))
        }
    }
}

#[cfg(feature = "serde")]
impl Serialize for SpecialEq<Arc<dyn RenameAliasFn>> {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        use serde::ser::Error;
        let mut buf = vec![];
        self.0
            .try_serialize(&mut buf)
            .map_err(|e| S::Error::custom(format!("{e}")))?;
        serializer.serialize_bytes(&buf)
    }
}

#[cfg(feature = "serde")]
impl<'a> Deserialize<'a> for SpecialEq<Arc<dyn RenameAliasFn>> {
    fn deserialize<D>(_deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'a>,
    {
        use serde::de::Error;
        Err(D::Error::custom(
            "deserialization not supported for this renaming function",
        ))
    }
}
