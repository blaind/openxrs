use std::{
    io::Cursor,
    iter,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::Duration,
};

use ash::{
    version::{DeviceV1_0, EntryV1_0, InstanceV1_0},
    vk::{self, Handle},
};
use gfx_backend_vulkan as back;
use gfx_hal::{command, device, format, image, memory, pass, pool, prelude::*, pso};
use openxr as xr;
use wgc::{
    instance::{RawAdapter, RawInstance},
    pipeline::ShaderModuleDescriptor,
};
use wgpu_core::instance::Instance;
use wgpu_types::BackendBit;

use wgpu_core as wgc;
use wgpu_types as wgt;
use wgt::ShaderFlags;

fn main() {
    env_logger::init();

    // Handle interrupts gracefully
    let running = Arc::new(AtomicBool::new(true));
    let r = running.clone();
    ctrlc::set_handler(move || {
        r.store(false, Ordering::Relaxed);
    })
    .expect("setting Ctrl-C handler");

    #[cfg(feature = "static")]
    let entry = xr::Entry::linked();
    #[cfg(not(feature = "static"))]
    let entry = xr::Entry::load()
        .expect("couldn't find the OpenXR loader; try enabling the \"static\" feature");

    // OpenXR will fail to initialize if we ask for an extension that OpenXR can't provide! So we
    // need to check all our extensions before initializing OpenXR with them. Note that even if the
    // extension is present, it's still possible you may not be able to use it. For example: the
    // hand tracking extension may be present, but the hand sensor might not be plugged in or turned
    // on. There are often additional checks that should be made before using certain features!
    let available_extensions = entry.enumerate_extensions().unwrap();

    // If a required extension isn't present, you want to ditch out here! It's possible something
    // like your rendering API might not be provided by the active runtime. APIs like OpenGL don't
    // have universal support.
    assert!(available_extensions.khr_vulkan_enable2);

    // Initialize OpenXR with the extensions we've found!
    let mut enabled_extensions = xr::ExtensionSet::default();
    enabled_extensions.khr_vulkan_enable2 = true;
    let xr_instance = entry
        .create_instance(
            &xr::ApplicationInfo {
                application_name: "openxrs example",
                application_version: 0,
                engine_name: "openxrs example",
                engine_version: 0,
            },
            &enabled_extensions,
            &[],
        )
        .unwrap();
    let instance_props = xr_instance.properties().unwrap();
    println!(
        "loaded OpenXR runtime: {} {}",
        instance_props.runtime_name, instance_props.runtime_version
    );

    // Request a form factor from the device (HMD, Handheld, etc.)
    let system = xr_instance
        .system(xr::FormFactor::HEAD_MOUNTED_DISPLAY)
        .unwrap();

    // Check what blend mode is valid for this device (opaque vs transparent displays). We'll just
    // take the first one available!
    let environment_blend_mode = xr_instance
        .enumerate_environment_blend_modes(system, VIEW_TYPE)
        .unwrap()[0];

    // OpenXR wants to ensure apps are using the correct graphics card and Vulkan features and
    // extensions, so the instance and device MUST be set up before Instance::create_session.

    let vk_target_version = vk::make_version(1, 1, 0); // Vulkan 1.1 guarantees multiview support
    let vk_target_version_xr = xr::Version::new(1, 1, 0);

    let reqs = xr_instance
        .graphics_requirements::<xr::Vulkan>(system)
        .unwrap();
    if vk_target_version_xr < reqs.min_api_version_supported
        || vk_target_version_xr.major() > reqs.max_api_version_supported.major()
    {
        panic!(
            "OpenXR runtime requires Vulkan version > {}, < {}.0.0",
            reqs.min_api_version_supported,
            reqs.max_api_version_supported.major() + 1
        );
    }

    let vk_entry = unsafe { ash::Entry::new().unwrap() };

    let vk_app_info = vk::ApplicationInfo::builder()
        .application_version(0)
        .engine_version(0)
        .api_version(vk_target_version);

    let extensions =
        back::Instance::required_extensions(&vk_entry, vk_target_version.into()).unwrap();
    let extensions_ptrs = extensions.iter().map(|&s| s.as_ptr()).collect::<Vec<_>>();

    let layers = back::Instance::required_layers(&vk_entry).unwrap();
    let layers_ptrs = layers.iter().map(|&s| s.as_ptr()).collect::<Vec<_>>();

    let vk_instance = unsafe {
        let vk_instance = xr_instance
            .create_vulkan_instance(
                system,
                std::mem::transmute(vk_entry.static_fn().get_instance_proc_addr),
                &vk::InstanceCreateInfo::builder()
                    .application_info(&vk_app_info)
                    .enabled_extension_names(&extensions_ptrs)
                    .enabled_layer_names(&layers_ptrs) as *const _ as *const _,
            )
            .expect("XR error creating Vulkan instance")
            .map_err(vk::Result::from_raw)
            .expect("Vulkan error creating Vulkan instance");
        ash::Instance::load(
            vk_entry.static_fn(),
            vk::Instance::from_raw(vk_instance as _),
        )
    };
    let vk_instance =
        drop_guard::DropGuard::new(vk_instance, |inst| unsafe { inst.destroy_instance(None) });

    let gfx_instance = unsafe {
        back::Instance::from_raw(
            vk_entry.clone(),
            vk_instance.clone(),
            vk_target_version.into(),
            extensions,
        )
        .unwrap()
    };

    let vk_physical_device = vk::PhysicalDevice::from_raw(
        xr_instance
            .vulkan_graphics_device(system, vk_instance.handle().as_raw() as _)
            .unwrap() as _,
    );

    let gfx_adapter = unsafe { gfx_instance.adapter_from_raw(vk_physical_device) };

    unsafe {
        let vk_device_properties = vk_instance.get_physical_device_properties(vk_physical_device);
        if vk_device_properties.api_version < vk_target_version {
            vk_instance.destroy_instance(None);
            panic!("Vulkan phyiscal device doesn't support version 1.1");
        }
    }

    let queue_family = gfx_adapter
        .queue_families
        .iter()
        .find(|family| family.queue_type().supports_graphics())
        .expect("Vulkan device has no graphics queue");

    let vk_device = unsafe {
        let vk_device = xr_instance
            .create_vulkan_device(
                system,
                std::mem::transmute(vk_entry.static_fn().get_instance_proc_addr),
                vk_physical_device.as_raw() as _,
                &vk::DeviceCreateInfo::builder().queue_create_infos(&[
                    vk::DeviceQueueCreateInfo::builder()
                        .queue_family_index(queue_family.id().0 as _)
                        .queue_priorities(&[1.0])
                        .build(),
                ]) as *const _ as *const _,
            )
            .expect("XR error creating Vulkan device")
            .map_err(vk::Result::from_raw)
            .expect("Vulkan error creating Vulkan device");
        ash::Device::load(vk_instance.fp_v1_0(), vk::Device::from_raw(vk_device as _))
    };
    let vk_device =
        drop_guard::DropGuard::new(vk_device, |dev| unsafe { dev.destroy_device(None) });

    // the arguments should mirror the ones for the creation of vk_device
    let mut gpu = unsafe {
        gfx_adapter
            .physical_device
            .gpu_from_raw(
                vk_device.clone(),
                &[(queue_family, &[1.0])],
                gfx_hal::Features::empty(), // add multiview when supported
            )
            .unwrap()
    };

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    let wgpu_instance = unsafe {
        wgc::hub::Global::from_raw(
            "openxr",
            IdentityPassThroughFactory,
            RawInstance::Vulkan(gfx_instance),
        )
    };

    let wgpu_adapter = unsafe {
        wgpu_instance.add_raw_adapter(
            // FIXME move input set into method?
            wgc::instance::AdapterInputs::IdSet(
                &[wgc::id::TypedId::zip(0, 0, wgt::Backend::Vulkan)],
                |id| id.backend(),
            ),
            RawAdapter::Vulkan(gfx_adapter),
        )
    }
    .unwrap();

    let supported_features =
        wgc::gfx_select!(wgpu_adapter => wgpu_instance.adapter_features(wgpu_adapter)).unwrap();

    println!("Features {:?}", supported_features);

    //let queue = &mut gpu.queue_groups.pop().unwrap().queues[0];

    println!("BACKEND: {:?}", wgpu_adapter.backend());

    let (wgpu_device, error) = unsafe {
        wgpu_instance.adapter_request_raw_device(
            wgpu_adapter,
            &wgt::DeviceDescriptor {
                label: None,
                features: wgt::Features::empty(),
                limits: wgt::Limits::default(),
            },
            None,
            wgc::id::TypedId::zip(0, 0, wgt::Backend::Vulkan),
            gpu,
        )
    };

    println!("DEVICE: {:?}", wgpu_device);

    let vert =
        gfx_auxil::read_spirv(Cursor::new(&include_bytes!("fullscreen.vert.spv")[..])).unwrap();
    let frag = gfx_auxil::read_spirv(Cursor::new(
        &include_bytes!("debug_pattern_single_view.frag.spv")[..],
    ))
    .unwrap();
    //let vert = unsafe { gpu.device.create_shader_module(&vert).unwrap() };
    //let frag = unsafe { gpu.device.create_shader_module(&frag).unwrap() };

    let desc = ShaderModuleDescriptor {
        label: None,
        flags: ShaderFlags::empty(), // VALIDATION ?
    };

    let (vert_shader_module_id, error) = wgpu_instance
        .device_create_shader_module::<gfx_backend_vulkan::Backend>(
            wgpu_device,
            &desc,
            wgc::pipeline::ShaderModuleSource::SpirV(Cow::from(vert)),
            wgc::id::TypedId::zip(0, 0, wgt::Backend::Vulkan),
        );

    let (frag_shader_module_id, error) = wgpu_instance
        .device_create_shader_module::<gfx_backend_vulkan::Backend>(
            wgpu_device,
            &desc,
            wgc::pipeline::ShaderModuleSource::SpirV(Cow::from(frag)),
            wgc::id::TypedId::zip(1, 0, wgt::Backend::Vulkan),
        );

    println!(
        "VERT: {:?}, FRAG: {:?}",
        vert_shader_module_id, frag_shader_module_id
    );
}

pub const COLOR_FORMAT: format::Format = format::Format::Bgra8Srgb;
pub const VK_COLOR_FORMAT: vk::Format = vk::Format::B8G8R8A8_SRGB;
const VIEW_TYPE: xr::ViewConfigurationType = xr::ViewConfigurationType::PRIMARY_STEREO;

struct Swapchain {
    handle: xr::Swapchain<xr::Vulkan>,
    buffers: Vec<Framebuffer>,
    resolution: vk::Extent2D,
}

struct Framebuffer {
    framebuffer: <back::Backend as gfx_hal::Backend>::Framebuffer,
    color: <back::Backend as gfx_hal::Backend>::ImageView,
}

/// Maximum number of frames in flight
const PIPELINE_DEPTH: usize = 2;

use std::{borrow::Cow, fmt::Debug, fs, marker::PhantomData, path::Path};

// TODO: just a copypaste below, sort out what's needed and whats not
#[derive(Debug)]
pub struct IdentityPassThrough<I>(PhantomData<I>);

impl<I: Clone + Debug + wgc::id::TypedId> wgc::hub::IdentityHandler<I> for IdentityPassThrough<I> {
    type Input = I;
    fn process(&self, id: I, backend: wgt::Backend) -> I {
        let (index, epoch, _backend) = id.unzip();
        I::zip(index, epoch, backend)
    }
    fn free(&self, _id: I) {}
}

pub struct IdentityPassThroughFactory;

impl<I: Clone + Debug + wgc::id::TypedId> wgc::hub::IdentityHandlerFactory<I>
    for IdentityPassThroughFactory
{
    type Filter = IdentityPassThrough<I>;
    fn spawn(&self, _min_index: u32) -> Self::Filter {
        IdentityPassThrough(PhantomData)
    }
}
impl wgc::hub::GlobalIdentityHandlerFactory for IdentityPassThroughFactory {}
