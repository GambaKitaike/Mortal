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

impl AgariDetail {
    /// Matches `preprocess_chips.chip_base`.
    #[must_use]
    pub const fn chip_base(&self) -> i8 {
        let mut base = self.num_aka as i8 + self.num_ura as i8;
        if self.ippatsu {
            base += 1;
        }
        if self.yakuman >= 1 {
            base += 5;
        }
        base
    }

    /// Matches `preprocess_chips.hora_chip_deltas`.
    #[must_use]
    pub fn chip_deltas(&self, actor: u8, target: u8) -> [i8; 4] {
        let base = self.chip_base();
        let mut deltas = [0i8; 4];
        if self.is_tsumo {
            for (i, d) in deltas.iter_mut().enumerate() {
                *d = if i as u8 == actor { base * 3 } else { -base };
            }
        } else {
            deltas[actor as usize] = base;
            deltas[target as usize] = -base;
        }
        deltas
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn chip_deltas_ron() {
        let detail = AgariDetail {
            point: 8000,
            fu: 30,
            han: 2,
            yakuman: 0,
            ippatsu: true,
            num_aka: 1,
            num_ura: 2,
            is_tsumo: false,
        };
        assert_eq!(detail.chip_base(), 4);
        assert_eq!(detail.chip_deltas(2, 1), [0, -4, 4, 0]);
    }

    #[test]
    fn chip_deltas_tsumo_yakuman() {
        let detail = AgariDetail {
            point: 48000,
            fu: 0,
            han: 0,
            yakuman: 1,
            ippatsu: false,
            num_aka: 0,
            num_ura: 0,
            is_tsumo: true,
        };
        assert_eq!(detail.chip_base(), 5);
        assert_eq!(detail.chip_deltas(0, 0), [15, -5, -5, -5]);
    }
}
