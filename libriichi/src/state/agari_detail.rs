use pyo3::prelude::*;

#[pyclass]
#[derive(Debug, Clone, Copy)]
pub struct AgariDetail {
    #[pyo3(get)]
    pub point: i32,
    #[pyo3(get)]
    pub fu: u8,
    #[pyo3(get)]
    pub han: u8,
    #[pyo3(get)]
    pub yakuman: u8,
    #[pyo3(get)]
    pub ippatsu: bool,
    #[pyo3(get)]
    pub num_aka: u8,
    #[pyo3(get)]
    pub num_ura: u8,
    #[pyo3(get)]
    pub is_tsumo: bool,
}
