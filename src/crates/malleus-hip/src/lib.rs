//! Safe wrappers over the subset of the HIP runtime used by Malleus. All `unsafe` lives here.

use std::ffi::{CStr, CString, c_char, c_int, c_uint, c_void};
use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

mod ffi {
    use super::*;
    pub type HipError = c_int;
    pub type Stream = *mut c_void;
    pub type Event = *mut c_void;
    pub type Module = *mut c_void;
    pub type Function = *mut c_void;
    pub type Graph = *mut c_void;
    pub type GraphExec = *mut c_void;
    pub type GraphNode = *mut c_void;

    pub const MEMCPY_H2D: c_int = 1;
    pub const MEMCPY_D2H: c_int = 2;
    pub const MEMCPY_D2D: c_int = 3;

    // hipStreamCaptureMode (only thread-local is used; capture stays confined to one stream+thread)
    pub const CAPTURE_MODE_THREAD_LOCAL: c_int = 1;

    // hipStreamCaptureStatus
    pub const CAPTURE_STATUS_NONE: c_int = 0;
    pub const CAPTURE_STATUS_ACTIVE: c_int = 1;

    #[link(name = "amdhip64")]
    unsafe extern "C" {
        pub fn hipInit(flags: c_uint) -> HipError;
        pub fn hipGetDeviceCount(n: *mut c_int) -> HipError;
        pub fn hipSetDevice(d: c_int) -> HipError;
        pub fn hipDeviceGetName(name: *mut c_char, len: c_int, d: c_int) -> HipError;
        pub fn hipGetDevicePropertiesR0600(props: *mut c_void, d: c_int) -> HipError;
        pub fn hipDeviceSynchronize() -> HipError;
        pub fn hipGetErrorString(e: HipError) -> *const c_char;
        pub fn hipMemGetInfo(free: *mut usize, total: *mut usize) -> HipError;
        pub fn hipMalloc(p: *mut *mut c_void, size: usize) -> HipError;
        pub fn hipFree(p: *mut c_void) -> HipError;
        pub fn hipHostMalloc(p: *mut *mut c_void, size: usize, flags: c_uint) -> HipError;
        pub fn hipHostFree(p: *mut c_void) -> HipError;
        pub fn hipMemcpy(dst: *mut c_void, src: *const c_void, size: usize, kind: c_int) -> HipError;
        pub fn hipMemcpyAsync(dst: *mut c_void, src: *const c_void, size: usize, kind: c_int, s: Stream) -> HipError;
        pub fn hipMemsetAsync(dst: *mut c_void, v: c_int, size: usize, s: Stream) -> HipError;
        pub fn hipStreamCreate(s: *mut Stream) -> HipError;
        pub fn hipStreamDestroy(s: Stream) -> HipError;
        pub fn hipStreamSynchronize(s: Stream) -> HipError;
        pub fn hipStreamBeginCapture(s: Stream, mode: c_int) -> HipError;
        pub fn hipStreamEndCapture(s: Stream, g: *mut Graph) -> HipError;
        pub fn hipStreamIsCapturing(s: Stream, status: *mut c_int) -> HipError;
        pub fn hipGraphInstantiateWithFlags(e: *mut GraphExec, g: Graph, flags: u64) -> HipError;
        pub fn hipGraphLaunch(e: GraphExec, s: Stream) -> HipError;
        pub fn hipGraphExecUpdate(e: GraphExec, g: Graph, err_node: *mut GraphNode, result: *mut c_int) -> HipError;
        pub fn hipGraphExecDestroy(e: GraphExec) -> HipError;
        pub fn hipGraphDestroy(g: Graph) -> HipError;
        pub fn hipEventCreate(e: *mut Event) -> HipError;
        pub fn hipEventDestroy(e: Event) -> HipError;
        pub fn hipEventRecord(e: Event, s: Stream) -> HipError;
        pub fn hipEventSynchronize(e: Event) -> HipError;
        pub fn hipEventElapsedTime(ms: *mut f32, a: Event, b: Event) -> HipError;
        pub fn hipModuleLoadData(m: *mut Module, image: *const c_void) -> HipError;
        pub fn hipModuleUnload(m: Module) -> HipError;
        pub fn hipModuleGetFunction(f: *mut Function, m: Module, name: *const c_char) -> HipError;
        pub fn hipModuleLaunchKernel(
            f: Function,
            gx: c_uint,
            gy: c_uint,
            gz: c_uint,
            bx: c_uint,
            by: c_uint,
            bz: c_uint,
            shared: c_uint,
            s: Stream,
            params: *mut *mut c_void,
            extra: *mut *mut c_void,
        ) -> HipError;
    }
}

#[derive(Debug, Clone)]
pub struct HipError {
    pub code: i32,
    pub what: String,
}

impl fmt::Display for HipError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // SAFETY: hipGetErrorString returns a static NUL-terminated string for any code.
        let msg = unsafe { CStr::from_ptr(ffi::hipGetErrorString(self.code)) };
        write!(
            f,
            "HIP error {} ({}) in {}",
            self.code,
            msg.to_string_lossy(),
            self.what
        )
    }
}

impl std::error::Error for HipError {}

pub type Result<T> = std::result::Result<T, HipError>;

fn check(code: ffi::HipError, what: &str) -> Result<()> {
    if code == 0 {
        Ok(())
    } else {
        Err(HipError {
            code,
            what: what.to_string(),
        })
    }
}

pub struct DeviceInfo {
    pub index: i32,
    pub name: String,
    pub arch: String,
}

/// Offset of `gcnArchName` in `hipDeviceProp_tR0600` (ROCm 6.x/7.x ABI).
const PROPS_SIZE: usize = 4096;
const GCN_ARCH_NAME_OFFSET: usize = 1160;

pub fn init() -> Result<()> {
    // SAFETY: plain FFI call with no pointers.
    check(unsafe { ffi::hipInit(0) }, "hipInit")
}

pub fn devices() -> Result<Vec<DeviceInfo>> {
    init()?;
    let mut n: c_int = 0;
    // SAFETY: n is a valid out pointer.
    check(unsafe { ffi::hipGetDeviceCount(&mut n) }, "hipGetDeviceCount")?;
    let mut out = Vec::new();
    for d in 0..n {
        let mut name = [0 as c_char; 256];
        // SAFETY: buffer length passed matches the buffer.
        check(
            unsafe { ffi::hipDeviceGetName(name.as_mut_ptr(), 256, d) },
            "hipDeviceGetName",
        )?;
        let mut props = vec![0u8; PROPS_SIZE];
        // SAFETY: props is larger than hipDeviceProp_tR0600.
        check(
            unsafe { ffi::hipGetDevicePropertiesR0600(props.as_mut_ptr().cast(), d) },
            "hipGetDeviceProperties",
        )?;
        let raw = &props[GCN_ARCH_NAME_OFFSET..GCN_ARCH_NAME_OFFSET + 256];
        let end = raw.iter().position(|&b| b == 0).unwrap_or(raw.len());
        let arch = String::from_utf8_lossy(&raw[..end]);
        out.push(DeviceInfo {
            index: d,
            // SAFETY: hipDeviceGetName NUL-terminates within the buffer.
            name: unsafe { CStr::from_ptr(name.as_ptr()) }.to_string_lossy().into_owned(),
            arch: arch.split(':').next().unwrap_or("").to_string(),
        });
    }
    Ok(out)
}

pub fn set_device(d: i32) -> Result<()> {
    // SAFETY: plain FFI call.
    check(unsafe { ffi::hipSetDevice(d) }, "hipSetDevice")
}

pub fn synchronize() -> Result<()> {
    // SAFETY: plain FFI call.
    check(unsafe { ffi::hipDeviceSynchronize() }, "hipDeviceSynchronize")
}

pub fn mem_info() -> Result<(usize, usize)> {
    let (mut f, mut t) = (0usize, 0usize);
    // SAFETY: valid out pointers.
    check(unsafe { ffi::hipMemGetInfo(&mut f, &mut t) }, "hipMemGetInfo")?;
    Ok((f, t))
}

/// Untyped device allocation.
pub struct DeviceBuffer {
    ptr: *mut c_void,
    bytes: usize,
    /// The allocation a window made by [`DeviceBuffer::into_views`] points into: it is freed when the last
    /// window goes, and dropping a window frees nothing by itself.
    parent: Option<Arc<DeviceBuffer>>,
}

// SAFETY: device pointers are plain addresses usable from any host thread.
unsafe impl Send for DeviceBuffer {}
// SAFETY: shared access only exposes the address; mutation goes through &mut or kernels.
unsafe impl Sync for DeviceBuffer {}

/// Live bytes the crate has allocated on the device, and the high-water mark since the process started.
/// On these APUs that memory is system RAM, so this is the number a memory pre-flight has to be checked
/// against once the model is loaded (`mem_get_info` reports the whole GTT pool, other processes included).
static LIVE_DEVICE_BYTES: AtomicUsize = AtomicUsize::new(0);
static PEAK_DEVICE_BYTES: AtomicUsize = AtomicUsize::new(0);

fn account(bytes: usize, add: bool) {
    let n = if add {
        LIVE_DEVICE_BYTES.fetch_add(bytes, Ordering::Relaxed) + bytes
    } else {
        LIVE_DEVICE_BYTES.fetch_sub(bytes, Ordering::Relaxed) - bytes
    };
    if add {
        PEAK_DEVICE_BYTES.fetch_max(n, Ordering::Relaxed);
    }
}

/// Bytes currently allocated on the device through this crate.
pub fn device_bytes_live() -> usize {
    LIVE_DEVICE_BYTES.load(Ordering::Relaxed)
}

/// Largest value `device_bytes_live` has had since the process started.
pub fn device_bytes_peak() -> usize {
    PEAK_DEVICE_BYTES.load(Ordering::Relaxed)
}

/// Where the device bytes went, when `MALLEUS_ALLOC_LEDGER=1`: every `DeviceBuffer::alloc` request, grouped by call
/// site, as (bytes, requests).
///
/// The pre-flight sizes a session from the config (`memplan::session_bytes`), and that estimate is only right if it
/// mirrors the allocator: this is what the allocator was actually asked for, so the two can be compared slot by slot
/// (issue #110: an Inkling-2 session measured ~3x its estimate at depth). Only the request is recorded, not where it
/// came from in the pool, so the ledger reads the same with `MALLEUS_BUF_POOL=0`.
/// Call site (`file:line`) to the (bytes, requests) it has asked for.
type Ledger = std::collections::HashMap<String, (u64, u64)>;

static LEDGER: std::sync::LazyLock<Option<std::sync::Mutex<Ledger>>> =
    std::sync::LazyLock::new(|| std::env::var_os("MALLEUS_ALLOC_LEDGER").map(|_| std::sync::Mutex::new(Ledger::new())));

fn ledger_note(bytes: usize, caller: &std::panic::Location<'_>) {
    let Some(l) = LEDGER.as_ref() else {
        return;
    };
    let key = format!("{}:{}", caller.file(), caller.line());
    let mut m = l.lock().unwrap();
    let e = m.entry(key).or_insert((0, 0));
    e.0 += bytes as u64;
    e.1 += 1;
}

/// The `MALLEUS_ALLOC_LEDGER` table so far: (call site, bytes, requests) by bytes, largest first.
pub fn alloc_ledger() -> Vec<(String, u64, u64)> {
    let Some(l) = LEDGER.as_ref() else {
        return Vec::new();
    };
    let m = l.lock().unwrap();
    let mut v: Vec<(String, u64, u64)> = m.iter().map(|(k, (b, n))| (k.clone(), *b, *n)).collect();
    v.sort_by_key(|(_, bytes, _)| std::cmp::Reverse(*bytes));
    v
}

/// Total bytes the ledger has recorded (0 when `MALLEUS_ALLOC_LEDGER` is unset): the difference between two calls is
/// what one stage of the run asked the allocator for.
pub fn alloc_ledger_bytes() -> u64 {
    alloc_ledger().iter().map(|(_, bytes, _)| *bytes).sum()
}

/// Prints [`alloc_ledger`] under a label, with the live device bytes: the pre-flight's estimate and this table are
/// the two halves of the same question.
pub fn print_alloc_ledger(label: &str) {
    if LEDGER.is_none() {
        return;
    }
    eprintln!(
        "alloc ledger ({label}): {:.2} GiB live on the device",
        device_bytes_live() as f64 / (1u64 << 30) as f64
    );
    for (site, bytes, n) in alloc_ledger() {
        eprintln!("  {:>10.2} MiB {:>7} x  {site}", bytes as f64 / (1u64 << 20) as f64, n);
    }
}

/// `hipMemGetInfo`: (free, total) of the device's allocatable memory, which on an APU is the GTT pool.
pub fn mem_get_info() -> Result<(usize, usize)> {
    let (mut free, mut total) = (0usize, 0usize);
    // SAFETY: both pointers are valid for a usize write.
    check(unsafe { ffi::hipMemGetInfo(&mut free, &mut total) }, "hipMemGetInfo")?;
    Ok((free, total))
}

/// Largest buffer the pool keeps, and how many bytes it holds at most.
const POOL_MAX_BUF: usize = 16 << 20;
const POOL_MAX_BYTES: usize = 512 << 20;

/// Freed small buffers kept for the next allocation of the same size.
///
/// A decode step allocates a few dozen short-lived buffers (logits, norms, the draft head's output), always of the
/// same sizes. Each `hipMalloc` costs ~100 us with the GPU idle and each `hipFree` synchronizes the device, which
/// measured as 2-3 ms per speculative round. Kernels and copies all run in stream order, so a buffer handed out
/// again is only written by work queued after its last reader.
///
/// `MALLEUS_BUF_POOL=0` turns it off (every buffer goes through `hipMalloc`/`hipFree`, as before).
struct Pool {
    /// Free addresses by exact size, with the tick of the last time the size was asked for or returned.
    free: std::collections::HashMap<usize, (Vec<usize>, u64)>,
    bytes: usize,
    tick: u64,
}

static POOL: std::sync::LazyLock<Option<std::sync::Mutex<Pool>>> = std::sync::LazyLock::new(|| {
    std::env::var("MALLEUS_BUF_POOL").map_or(true, |v| v != "0").then(|| {
        std::sync::Mutex::new(Pool {
            free: std::collections::HashMap::new(),
            bytes: 0,
            tick: 0,
        })
    })
});

fn pool_take(bytes: usize) -> Option<*mut c_void> {
    if bytes > POOL_MAX_BUF {
        return None;
    }
    let mut p = POOL.as_ref()?.lock().unwrap();
    p.tick += 1;
    let tick = p.tick;
    let (list, used) = p.free.get_mut(&bytes)?;
    *used = tick;
    let ptr = list.pop()?;
    p.bytes -= bytes;
    Some(ptr as *mut c_void)
}

/// Keeps `ptr` for reuse; false when the caller has to free it. Makes room by dropping the sizes used least
/// recently, so the sizes of a decode step stay pooled across prompts of other lengths.
fn pool_put(ptr: *mut c_void, bytes: usize) -> bool {
    if bytes > POOL_MAX_BUF {
        return false;
    }
    let Some(pool) = POOL.as_ref() else {
        return false;
    };
    let mut evicted = Vec::new();
    {
        let mut p = pool.lock().unwrap();
        p.tick += 1;
        let tick = p.tick;
        while p.bytes + bytes > POOL_MAX_BYTES {
            let Some(old) = p
                .free
                .iter()
                .filter(|(size, (list, _))| **size != bytes && !list.is_empty())
                .min_by_key(|(_, (_, used))| *used)
                .map(|(size, _)| *size)
            else {
                break;
            };
            let (list, _) = p.free.remove(&old).unwrap();
            p.bytes -= old * list.len();
            evicted.extend(list.into_iter().map(|a| (a, old)));
        }
        if p.bytes + bytes > POOL_MAX_BYTES {
            drop(p);
            free_all(evicted);
            return false;
        }
        let e = p.free.entry(bytes).or_insert((Vec::new(), tick));
        e.0.push(ptr as usize);
        e.1 = tick;
        p.bytes += bytes;
    }
    free_all(evicted);
    true
}

fn free_all(list: Vec<(usize, usize)>) {
    for (addr, bytes) in list {
        // SAFETY: the address came from hipMalloc and left the pool, so nothing else refers to it.
        unsafe { ffi::hipFree(addr as *mut c_void) };
        account(bytes, false);
    }
}

impl DeviceBuffer {
    #[track_caller]
    pub fn alloc(bytes: usize) -> Result<Self> {
        ledger_note(bytes, std::panic::Location::caller());
        let size = bytes.max(4);
        if let Some(ptr) = pool_take(size) {
            return Ok(DeviceBuffer {
                ptr,
                bytes,
                parent: None,
            });
        }
        let mut ptr = std::ptr::null_mut();
        // SAFETY: valid out pointer; size 0 is rounded up to keep a unique address.
        check(
            unsafe { ffi::hipMalloc(&mut ptr, size) },
            &format!("hipMalloc({bytes})"),
        )?;
        account(size, true);
        Ok(DeviceBuffer {
            ptr,
            bytes,
            parent: None,
        })
    }

    /// Turns the buffer into windows `(offset, len)` that share it: the allocation lives until the last one is
    /// dropped. Several small inputs of a step can then be uploaded with one host copy through a window that
    /// covers them all.
    pub fn into_views(self, parts: &[(usize, usize)]) -> Vec<DeviceBuffer> {
        let whole = Arc::new(self);
        parts
            .iter()
            .map(|&(offset, len)| {
                assert!(offset + len <= whole.bytes, "view out of the buffer");
                DeviceBuffer {
                    // SAFETY: inside the allocation (checked above), which `parent` keeps alive.
                    ptr: unsafe { whole.ptr.byte_add(offset) },
                    bytes: len,
                    parent: Some(whole.clone()),
                }
            })
            .collect()
    }

    #[track_caller]
    pub fn zeroed(bytes: usize) -> Result<Self> {
        let b = Self::alloc(bytes)?;
        b.memset(0, &Stream::null())?;
        Stream::null().synchronize()?;
        Ok(b)
    }

    #[track_caller]
    pub fn from_slice<T: Copy>(data: &[T]) -> Result<Self> {
        let mut b = Self::alloc(std::mem::size_of_val(data))?;
        b.upload(data)?;
        Ok(b)
    }

    pub fn len(&self) -> usize {
        self.bytes
    }

    pub fn is_empty(&self) -> bool {
        self.bytes == 0
    }

    pub fn addr(&self) -> u64 {
        self.ptr as u64
    }

    pub fn upload<T: Copy>(&mut self, data: &[T]) -> Result<()> {
        self.upload_at(0, data)
    }

    pub fn upload_at<T: Copy>(&mut self, offset: usize, data: &[T]) -> Result<()> {
        let n = std::mem::size_of_val(data);
        assert!(offset + n <= self.bytes, "upload out of bounds");
        // SAFETY: bounds checked above; source is a valid host slice.
        check(
            unsafe {
                ffi::hipMemcpy(
                    self.ptr.cast::<u8>().add(offset).cast(),
                    data.as_ptr().cast(),
                    n,
                    ffi::MEMCPY_H2D,
                )
            },
            "hipMemcpy H2D",
        )
    }

    pub fn download<T: Copy + Default>(&self, n: usize) -> Result<Vec<T>> {
        self.download_at(0, n)
    }

    pub fn download_at<T: Copy + Default>(&self, offset: usize, n: usize) -> Result<Vec<T>> {
        let bytes = n * std::mem::size_of::<T>();
        assert!(offset + bytes <= self.bytes, "download out of bounds");
        let mut v = vec![T::default(); n];
        // SAFETY: bounds checked above; destination has exactly `bytes` bytes.
        check(
            unsafe {
                ffi::hipMemcpy(
                    v.as_mut_ptr().cast(),
                    self.ptr.cast::<u8>().add(offset).cast(),
                    bytes,
                    ffi::MEMCPY_D2H,
                )
            },
            "hipMemcpy D2H",
        )?;
        Ok(v)
    }

    pub fn copy_from(
        &mut self,
        dst_off: usize,
        src: &DeviceBuffer,
        src_off: usize,
        bytes: usize,
        s: &Stream,
    ) -> Result<()> {
        self.copy_from_shared(dst_off, src, src_off, bytes, s)
    }

    /// The same copy with a shared receiver, for destinations reached through `&DeviceBuffer`.
    pub fn copy_from_shared(
        &self,
        dst_off: usize,
        src: &DeviceBuffer,
        src_off: usize,
        bytes: usize,
        s: &Stream,
    ) -> Result<()> {
        assert!(
            dst_off + bytes <= self.bytes && src_off + bytes <= src.bytes,
            "d2d out of bounds"
        );
        // SAFETY: bounds checked above.
        check(
            unsafe {
                ffi::hipMemcpyAsync(
                    self.ptr.cast::<u8>().add(dst_off).cast(),
                    src.ptr.cast::<u8>().add(src_off).cast(),
                    bytes,
                    ffi::MEMCPY_D2D,
                    s.raw,
                )
            },
            "hipMemcpy D2D",
        )
    }

    pub fn memset(&self, v: u8, s: &Stream) -> Result<()> {
        // SAFETY: whole allocation.
        check(
            unsafe { ffi::hipMemsetAsync(self.ptr, v as c_int, self.bytes, s.raw) },
            "hipMemset",
        )
    }

    /// Sets the first `bytes` bytes (a scratch buffer sized for the largest input but used for fewer rows).
    pub fn memset_prefix(&self, v: u8, bytes: usize, s: &Stream) -> Result<()> {
        if bytes > self.bytes {
            // hipErrorInvalidValue
            return Err(HipError {
                code: 1,
                what: format!("memset of {bytes} bytes on a {}-byte buffer", self.bytes),
            });
        }
        // SAFETY: bytes is within the allocation.
        check(
            unsafe { ffi::hipMemsetAsync(self.ptr, v as c_int, bytes, s.raw) },
            "hipMemset",
        )
    }
}

impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        if self.parent.is_some() || pool_put(self.ptr, self.bytes.max(4)) {
            return;
        }
        // SAFETY: ptr came from hipMalloc and is freed once.
        unsafe { ffi::hipFree(self.ptr) };
        account(self.bytes.max(4), false);
    }
}

/// Access hints for files the engine reads while it runs (`posix_fadvise`). They live here because this is the
/// crate that is allowed to call into C.
pub mod file_hint {
    use std::ffi::c_int;
    use std::os::fd::AsRawFd;

    unsafe extern "C" {
        /// `posix_fadvise(2)`; `off_t` is 64 bits on the targets this engine builds for.
        fn posix_fadvise(fd: c_int, offset: i64, len: i64, advice: c_int) -> c_int;
    }
    const POSIX_FADV_RANDOM: c_int = 1;
    const POSIX_FADV_WILLNEED: c_int = 3;

    /// The file is read in small pieces at unrelated offsets: no readahead window around each read.
    pub fn random(file: &std::fs::File) {
        // SAFETY: a valid descriptor; the call only sets a readahead hint.
        unsafe { posix_fadvise(file.as_raw_fd(), 0, 0, POSIX_FADV_RANDOM) };
    }

    /// Starts reading `len` bytes at `offset` into the page cache without waiting for them.
    pub fn will_need(file: &std::fs::File, offset: u64, len: u64) {
        // SAFETY: a valid descriptor; the call only starts readahead.
        unsafe { posix_fadvise(file.as_raw_fd(), offset as i64, len as i64, POSIX_FADV_WILLNEED) };
    }
}

/// Page-locked host memory for fast uploads.
pub struct PinnedBuffer {
    ptr: *mut u8,
    bytes: usize,
}

// SAFETY: owned host allocation.
unsafe impl Send for PinnedBuffer {}

impl PinnedBuffer {
    pub fn alloc(bytes: usize) -> Result<Self> {
        let mut ptr = std::ptr::null_mut();
        // SAFETY: valid out pointer.
        check(
            unsafe { ffi::hipHostMalloc(&mut ptr, bytes.max(4), 0) },
            "hipHostMalloc",
        )?;
        Ok(PinnedBuffer { ptr: ptr.cast(), bytes })
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: ptr is a live allocation of `bytes` bytes owned by self.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.bytes) }
    }

    /// Copies `len` bytes from this buffer into `dst` at `dst_off`, synchronously.
    pub fn copy_to(&self, len: usize, dst: &mut DeviceBuffer, dst_off: usize) -> Result<()> {
        assert!(
            len <= self.bytes && dst_off + len <= dst.bytes,
            "pinned copy out of bounds"
        );
        // SAFETY: bounds checked above.
        check(
            unsafe {
                ffi::hipMemcpy(
                    dst.ptr.cast::<u8>().add(dst_off).cast(),
                    self.ptr.cast(),
                    len,
                    ffi::MEMCPY_H2D,
                )
            },
            "hipMemcpy pinned H2D",
        )
    }
}

impl Drop for PinnedBuffer {
    fn drop(&mut self) {
        // SAFETY: ptr came from hipHostMalloc.
        unsafe { ffi::hipHostFree(self.ptr.cast()) };
    }
}

pub struct Stream {
    raw: ffi::Stream,
    owned: bool,
}

// SAFETY: HIP streams are thread-safe handles.
unsafe impl Send for Stream {}
// SAFETY: HIP streams are thread-safe handles.
unsafe impl Sync for Stream {}

impl Stream {
    pub fn new() -> Result<Self> {
        let mut raw = std::ptr::null_mut();
        // SAFETY: valid out pointer.
        check(unsafe { ffi::hipStreamCreate(&mut raw) }, "hipStreamCreate")?;
        Ok(Stream { raw, owned: true })
    }

    pub fn null() -> Self {
        Stream {
            raw: std::ptr::null_mut(),
            owned: false,
        }
    }

    pub fn synchronize(&self) -> Result<()> {
        // SAFETY: raw is a valid stream or null (default stream).
        check(unsafe { ffi::hipStreamSynchronize(self.raw) }, "hipStreamSynchronize")
    }

    /// Begins stream capture: work launched on this stream is recorded into a graph instead of
    /// executing, until [`Stream::end_capture`]. Capture does not work on the default (null) stream,
    /// so `self` must have been made with [`Stream::new`]. Uses thread-local capture mode, so only
    /// work on this stream (on this thread) is captured and unrelated streams are unaffected.
    pub fn begin_capture(&self) -> Result<()> {
        // SAFETY: valid owned stream; mode is a documented enum value.
        check(
            unsafe { ffi::hipStreamBeginCapture(self.raw, ffi::CAPTURE_MODE_THREAD_LOCAL) },
            "hipStreamBeginCapture",
        )
    }

    /// Ends capture started by [`Stream::begin_capture`] and returns the recorded graph.
    pub fn end_capture(&self) -> Result<Graph> {
        let mut raw = std::ptr::null_mut();
        // SAFETY: valid stream in the capturing state; out pointer is valid.
        check(
            unsafe { ffi::hipStreamEndCapture(self.raw, &mut raw) },
            "hipStreamEndCapture",
        )?;
        Ok(Graph { raw })
    }

    /// Whether this stream is currently capturing.
    pub fn is_capturing(&self) -> Result<bool> {
        let mut status = ffi::CAPTURE_STATUS_NONE;
        // SAFETY: valid stream and out pointer.
        check(
            unsafe { ffi::hipStreamIsCapturing(self.raw, &mut status) },
            "hipStreamIsCapturing",
        )?;
        Ok(status == ffi::CAPTURE_STATUS_ACTIVE)
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        if self.owned {
            // SAFETY: stream created by hipStreamCreate.
            unsafe { ffi::hipStreamDestroy(self.raw) };
        }
    }
}

/// A captured HIP graph: the recorded topology of kernels/copies, not yet runnable. Instantiate it
/// into a [`GraphExec`] to run it. Produced by [`Stream::end_capture`].
pub struct Graph {
    raw: ffi::Graph,
}

// SAFETY: graph handles are opaque and only moved between threads, never aliased for mutation.
unsafe impl Send for Graph {}

impl Graph {
    /// Instantiates the graph into an executable form. This does the expensive work (node setup,
    /// validation) once; the resulting [`GraphExec`] can then be launched many times cheaply.
    pub fn instantiate(&self) -> Result<GraphExec> {
        let mut raw = std::ptr::null_mut();
        // SAFETY: valid graph; out pointer is valid; flags 0 is the default instantiation.
        check(
            unsafe { ffi::hipGraphInstantiateWithFlags(&mut raw, self.raw, 0) },
            "hipGraphInstantiateWithFlags",
        )?;
        Ok(GraphExec { raw })
    }

    fn raw(&self) -> ffi::Graph {
        self.raw
    }
}

impl Drop for Graph {
    fn drop(&mut self) {
        // SAFETY: graph produced by hipStreamEndCapture.
        unsafe { ffi::hipGraphDestroy(self.raw) };
    }
}

/// An instantiated, runnable HIP graph. Launching it replays the whole recorded sequence with a
/// single runtime submission, which removes the per-kernel launch gap that dominates decode.
pub struct GraphExec {
    raw: ffi::GraphExec,
}

// SAFETY: exec handles are opaque; launching takes `&self` but concurrent launches of the same exec
// are the caller's responsibility (HIP does not allow it), so we expose Send only, not Sync.
unsafe impl Send for GraphExec {}

impl GraphExec {
    /// Replays the graph on `stream`. The stream may be the default stream.
    pub fn launch(&self, stream: &Stream) -> Result<()> {
        // SAFETY: valid exec and stream.
        check(unsafe { ffi::hipGraphLaunch(self.raw, stream.raw) }, "hipGraphLaunch")
    }

    /// Tries to patch this executable in place to match a freshly captured `graph` with the same
    /// topology but different node parameters (e.g. a changed launch immediate). Returns `Ok(true)`
    /// if the update succeeded, or `Ok(false)` if HIP could not patch it (topology differs, or the
    /// change is unsupported) — in that case the caller must re-instantiate. Patching is far cheaper
    /// than re-instantiating, so it is the fast path when only scalar arguments move between steps.
    pub fn update(&self, graph: &Graph) -> Result<bool> {
        let mut err_node = std::ptr::null_mut();
        let mut result = 0;
        // SAFETY: valid exec and graph; both out pointers are valid.
        let code = unsafe { ffi::hipGraphExecUpdate(self.raw, graph.raw(), &mut err_node, &mut result) };
        Ok(code == 0)
    }
}

impl Drop for GraphExec {
    fn drop(&mut self) {
        // SAFETY: exec produced by hipGraphInstantiateWithFlags.
        unsafe { ffi::hipGraphExecDestroy(self.raw) };
    }
}

pub struct Event {
    raw: ffi::Event,
}

impl Event {
    pub fn new() -> Result<Self> {
        let mut raw = std::ptr::null_mut();
        // SAFETY: valid out pointer.
        check(unsafe { ffi::hipEventCreate(&mut raw) }, "hipEventCreate")?;
        Ok(Event { raw })
    }

    pub fn record(&self, s: &Stream) -> Result<()> {
        // SAFETY: valid event and stream.
        check(unsafe { ffi::hipEventRecord(self.raw, s.raw) }, "hipEventRecord")
    }

    pub fn elapsed_ms(&self, start: &Event) -> Result<f32> {
        let mut ms = 0f32;
        // SAFETY: valid events.
        check(unsafe { ffi::hipEventSynchronize(self.raw) }, "hipEventSynchronize")?;
        // SAFETY: valid events and out pointer.
        check(
            unsafe { ffi::hipEventElapsedTime(&mut ms, start.raw, self.raw) },
            "hipEventElapsedTime",
        )?;
        Ok(ms)
    }
}

impl Drop for Event {
    fn drop(&mut self) {
        // SAFETY: event created by hipEventCreate.
        unsafe { ffi::hipEventDestroy(self.raw) };
    }
}

pub struct Module {
    raw: ffi::Module,
}

// SAFETY: module handles are process-global and thread-safe.
unsafe impl Send for Module {}
// SAFETY: module handles are process-global and thread-safe.
unsafe impl Sync for Module {}

impl Module {
    pub fn load(image: &[u8]) -> Result<Arc<Self>> {
        let mut raw = std::ptr::null_mut();
        // SAFETY: image is a code object kept alive for the duration of the call (HIP copies it).
        check(
            unsafe { ffi::hipModuleLoadData(&mut raw, image.as_ptr().cast()) },
            "hipModuleLoadData",
        )?;
        Ok(Arc::new(Module { raw }))
    }

    pub fn function(self: &Arc<Self>, name: &str) -> Result<Kernel> {
        let c = CString::new(name).expect("kernel name");
        let mut f = std::ptr::null_mut();
        // SAFETY: valid module and NUL-terminated name.
        check(
            unsafe { ffi::hipModuleGetFunction(&mut f, self.raw, c.as_ptr()) },
            &format!("hipModuleGetFunction({name})"),
        )?;
        Ok(Kernel {
            raw: f,
            name: name.to_string(),
            _m: Arc::clone(self),
        })
    }
}

impl Drop for Module {
    fn drop(&mut self) {
        // SAFETY: module loaded by hipModuleLoadData.
        unsafe { ffi::hipModuleUnload(self.raw) };
    }
}

#[derive(Clone, Copy, Debug)]
pub enum Arg {
    Ptr(u64),
    I32(i32),
    U32(u32),
    F32(f32),
    U64(u64),
}

impl From<&DeviceBuffer> for Arg {
    fn from(b: &DeviceBuffer) -> Self {
        Arg::Ptr(b.addr())
    }
}
impl From<&mut DeviceBuffer> for Arg {
    fn from(b: &mut DeviceBuffer) -> Self {
        Arg::Ptr(b.addr())
    }
}
impl From<i32> for Arg {
    fn from(v: i32) -> Self {
        Arg::I32(v)
    }
}
impl From<u32> for Arg {
    fn from(v: u32) -> Self {
        Arg::U32(v)
    }
}
impl From<f32> for Arg {
    fn from(v: f32) -> Self {
        Arg::F32(v)
    }
}
impl From<usize> for Arg {
    fn from(v: usize) -> Self {
        Arg::I32(i32::try_from(v).expect("kernel int arg overflow"))
    }
}

pub struct Kernel {
    raw: ffi::Function,
    name: String,
    _m: Arc<Module>,
}

// SAFETY: function handles are immutable after lookup.
unsafe impl Send for Kernel {}
// SAFETY: function handles are immutable after lookup.
unsafe impl Sync for Kernel {}

impl Kernel {
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Launches the kernel. Pointer args must reference live allocations sized for the kernel's accesses.
    pub fn launch(
        &self,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        shared: u32,
        s: &Stream,
        args: &[Arg],
    ) -> Result<()> {
        let mut slots: Vec<u64> = args
            .iter()
            .map(|a| match *a {
                Arg::Ptr(p) | Arg::U64(p) => p,
                Arg::I32(v) => v as u32 as u64,
                Arg::U32(v) => v as u64,
                Arg::F32(v) => v.to_bits() as u64,
            })
            .collect();
        let mut ptrs: Vec<*mut c_void> = slots.iter_mut().map(|s| (s as *mut u64).cast()).collect();
        // SAFETY: each param pointer refers to a little-endian slot at least as large as the argument.
        check(
            unsafe {
                ffi::hipModuleLaunchKernel(
                    self.raw,
                    grid.0,
                    grid.1,
                    grid.2,
                    block.0,
                    block.1,
                    block.2,
                    shared,
                    s.raw,
                    ptrs.as_mut_ptr(),
                    std::ptr::null_mut(),
                )
            },
            &self.name,
        )
    }
}
