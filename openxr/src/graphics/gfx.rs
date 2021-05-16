use std::{marker::PhantomData, ptr};

use crate::Instance;
use gfx_hal::Backend;
use sys::platform::*;

use crate::*;

/// The GFX graphics API, that wraps underlying Vulkan / OpenGL API
pub struct Gfx<B: Backend> {
    _phantom: PhantomData<B>,
}

impl<B: Backend> Graphics for Gfx<B> {
    type Requirements = Requirements;
    type Format = VkFormat;
    type SessionCreateInfo = SessionCreateInfo<B>;
    type SwapchainImage = VkImage;

    fn raise_format(x: i64) -> Self::Format {
        x as _
    }
    fn lower_format(x: Self::Format) -> i64 {
        x as _
    }

    fn requirements(instance: &Instance, system: SystemId) -> Result<Requirements> {
        let out = unsafe {
            let mut x = sys::GraphicsRequirementsVulkanKHR::out(ptr::null_mut());
            cvt((instance.vulkan().get_vulkan_graphics_requirements2)(
                instance.as_raw(),
                system,
                x.as_mut_ptr(),
            ))?;
            x.assume_init()
        };
        Ok(Requirements {
            min_api_version_supported: out.min_api_version_supported,
            max_api_version_supported: out.max_api_version_supported,
        })
    }

    unsafe fn create_session(
        instance: &Instance,
        system: SystemId,
        info: &Self::SessionCreateInfo,
    ) -> Result<sys::Session> {
        let binding = sys::GraphicsBindingVulkanKHR {
            ty: sys::GraphicsBindingVulkanKHR::TYPE,
            next: ptr::null(),
            instance: info.instance, // FIXME COOP: pub trait Instance<B: Backend> needs a raw getter?
            physical_device: info.physical_device, // FIXME COOP: pub trait PhysicalDevice<B: Backend> needs a raw getter?
            device: info.device, // FIXME COOP: pub trait Device<B: Backend> needs a raw getter?
            queue_family_index: info.queue_family_index, // FIXME TODO
            queue_index: info.queue_index, // FIXME TODO
        };
        let info = sys::SessionCreateInfo {
            ty: sys::SessionCreateInfo::TYPE,
            next: &binding as *const _ as *const _,
            create_flags: Default::default(),
            system_id: system,
        };
        let mut out = sys::Session::NULL;
        cvt((instance.fp().create_session)(
            instance.as_raw(),
            &info,
            &mut out,
        ))?;
        Ok(out)
    }

    fn enumerate_swapchain_images(
        swapchain: &Swapchain<Self>,
    ) -> Result<Vec<Self::SwapchainImage>> {
        let images = get_arr_init(
            sys::SwapchainImageVulkanKHR {
                ty: sys::SwapchainImageVulkanKHR::TYPE,
                next: ptr::null_mut(),
                image: 0,
            },
            |capacity, count, buf| unsafe {
                (swapchain.instance().fp().enumerate_swapchain_images)(
                    swapchain.as_raw(),
                    capacity,
                    count,
                    buf as *mut _,
                )
            },
        )?;
        Ok(images.into_iter().map(|x| x.image as _).collect())
    }
}

#[derive(Copy, Clone)]
pub struct Requirements {
    pub min_api_version_supported: Version,
    pub max_api_version_supported: Version,
}

#[derive(Clone)]
pub struct SessionCreateInfo<B: Backend> {
    pub instance: B::Instance,
    pub physical_device: B::PhysicalDevice,
    pub device: B::Device,
    pub queue: B::Queue,
}
