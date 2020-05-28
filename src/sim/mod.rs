pub mod memory;
pub mod pipeline;
pub use memory::MemoryModel;
pub use pipeline::PipelineModel;

use ro_cell::RoCell;

type PipelineFactory = fn(usize) -> Box<dyn pipeline::PipelineModel>;

// As in register_memory_model will use Box::from_raw, we must use a ZST type to avoid freeing underlying memory.
static MEMORY_MODEL: RoCell<&'static dyn MemoryModel> = RoCell::new(&memory::AtomicModel);
static PIPELINE_MODEL: RoCell<PipelineFactory> = RoCell::new(|_| Box::new(pipeline::AtomicModel));

pub fn get_memory_model() -> &'static dyn MemoryModel {
    *MEMORY_MODEL
}

pub fn new_pipeline_model(hartid: usize) -> Box<dyn PipelineModel> {
    (*PIPELINE_MODEL)(hartid)
}

unsafe fn register_memory_model(model: Box<dyn MemoryModel>) -> Box<dyn MemoryModel> {
    Box::from_raw(RoCell::replace(&MEMORY_MODEL, Box::leak(model)) as *const dyn MemoryModel as _)
}

unsafe fn register_pipeline_model(model: PipelineFactory) -> PipelineFactory {
    RoCell::replace(&PIPELINE_MODEL, model)
}

/// Set whether lock-step execution is required for this model's simulation.
/// For cycle-level simulation you would want this to be true, but if no cache coherency is
/// simulated **and** only rough metrics are needed it's okay to set it to false.
unsafe fn set_lockstep_mode(mode: bool) {
    RoCell::as_mut(&crate::FLAGS).thread = !mode;
}

pub unsafe fn switch_model(id: usize) {
    match id {
        0 => {
            register_memory_model(Box::new(memory::AtomicModel));
            register_pipeline_model(|_| Box::new(pipeline::AtomicModel));
            set_lockstep_mode(false);
        }
        1 => {
            register_memory_model(Box::new(memory::SimpleModel));
            register_pipeline_model(|_| Box::new(pipeline::InOrderModel::default()));
            set_lockstep_mode(false);
        }
        _ => panic!("unknown model id"),
    }
}
