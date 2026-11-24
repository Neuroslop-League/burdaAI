// serde_json::Value normally discards earlier values of duplicate object keys.
use serde::de::{self, Deserialize, Deserializer, MapAccess, SeqAccess, Visitor};
use serde_json::{Map, Value};
use std::fmt;

struct Unique(Value);
struct ValueVisitor;

impl<'de> Deserialize<'de> for Unique {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        d.deserialize_any(ValueVisitor)
    }
}

impl<'de> Visitor<'de> for ValueVisitor {
    type Value = Unique;
    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("JSON with unique keys")
    }
    fn visit_bool<E: de::Error>(self, x: bool) -> Result<Unique, E> {
        Ok(Unique(x.into()))
    }
    fn visit_i64<E: de::Error>(self, x: i64) -> Result<Unique, E> {
        Ok(Unique(x.into()))
    }
    fn visit_u64<E: de::Error>(self, x: u64) -> Result<Unique, E> {
        Ok(Unique(x.into()))
    }
    fn visit_f64<E: de::Error>(self, x: f64) -> Result<Unique, E> {
        Ok(Unique(x.into()))
    }
    fn visit_str<E: de::Error>(self, x: &str) -> Result<Unique, E> {
        Ok(Unique(x.into()))
    }
    fn visit_unit<E: de::Error>(self) -> Result<Unique, E> {
        Ok(Unique(Value::Null))
    }
    fn visit_seq<A: SeqAccess<'de>>(self, mut a: A) -> Result<Unique, A::Error> {
        let mut values = Vec::new();
        while let Some(Unique(value)) = a.next_element()? {
            values.push(value);
        }
        Ok(Unique(Value::Array(values)))
    }
    fn visit_map<A: MapAccess<'de>>(self, mut a: A) -> Result<Unique, A::Error> {
        let mut object = Map::new();
        while let Some((key, Unique(value))) = a.next_entry::<String, Unique>()? {
            if object.contains_key(&key) {
                return Err(de::Error::custom(format!("duplicate key: {key}")));
            }
            object.insert(key, value);
        }
        Ok(Unique(Value::Object(object)))
    }
}

pub fn parse(bytes: &[u8]) -> Result<Value, serde_json::Error> {
    Ok(serde_json::from_slice::<Unique>(bytes)?.0)
}
