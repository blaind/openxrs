use std::ptr;

use crate::Instance;
use ash::{version::InstanceV1_0, vk::Handle};
use gfx_backend_vulkan::{Device, Instance as GfxVulkanInstance};
use gfx_hal::queue::QueueFamilyId;
use sys::platform::*;

use crate::*;

/// The GFX graphics API, that wraps underlying Vulkan / OpenGL API
pub struct Gfx;

impl Graphics for Gfx {
    type Requirements = Requirements;
    type Format = VkFormat;
    type SessionCreateInfo = SessionCreateInfo;
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
            instance: info.instance as *const _,
            physical_device: info.physical_device as *const _,
            device: info.device as *const _,
            queue_family_index: info.queue_family.0 as u32,
            queue_index: info.queue_id,
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

pub struct SessionCreateInfo {
    pub instance: u64,
    pub physical_device: u64,
    pub device: u64,
    pub queue_family: QueueFamilyId,
    pub queue_id: u32,
}
