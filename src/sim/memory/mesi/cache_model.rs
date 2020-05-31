//! Constants for modelling
#![allow(dead_code)]

/// Net delay between L1 and L2 cache
pub const L1_L2_NET_DELAY: usize = 2;

/// TLB delay between TLB miss and walk start
pub const TLB_WALK_START_DELAY: usize = 1;

/// TLB delay between page walk and finish
pub const TLB_WALK_END_DELAY: usize = 1;

/// TLB delay for sending request to L2
/// We have 0 here because we assume TLB can use replied address immediately
/// for next level walk.
pub const TLB_GET_DELAY: usize = 0;

/// TLB delay after receiving a Get-Ack from L2.
pub const TLB_GET_ACK_DELAY: usize = 1;

// #region L2 delays
//

/// L2 delay after receiving a Get from L1
pub const L2_GET_DELAY: usize = 1;

// /// L2 delay before sending a Get-Ack to L1
// pub const L2_GET_ACK_DELAY: usize = 0;

/// L2 delay before sending Inv
pub const L2_INV_DELAY: usize = 0;

/// L2 delay after receving Inv-Ack
pub const L2_INV_ACK_DELAY: usize = 1;

/// L2 delay between sending to and receiving from main memory
pub const MEM_DELAY: usize = 10;

//
// #endregion

// #region L1 delays
//

/// L1 delay before sending Get to L2
pub const L1_GET_DELAY: usize = 1;

/// L1 delay after receving a Get-Ack from L2
pub const L1_GET_ACK_DELAY: usize = 1;

/// L1 delay between receving Inv and sending Inv-Ack
pub const L1_INV_DELAY: usize = 1;

//
// #endregion
