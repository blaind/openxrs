//! Illustrates rendering using Vulkan with multiview. Supports any Vulkan 1.1 capable environment.
//!
//! Renders a smooth gradient across the entire view, with different colors per eye.
//!
//! This example uses minimal abstraction for clarity. Real-world code should encapsulate and
//! largely decouple its Vulkan and OpenXR components and handle errors gracefully.

use std::{
    io::Cursor,
    iter,
    sync::{atomic, Arc},
    time::Duration,
};

use ash::{
    util::read_spv,
    version::EntryV1_0,
    vk::{self, Handle},
};

use libc::c_void;
use openxr as xr;

use gfx_backend_vulkan as back;
use gfx_hal::{command, device, format, image, pass, pool, prelude::*, pso, RawInstance};

#[allow(clippy::field_reassign_with_default)] // False positive, might be fixed 1.51
fn main() {
    // Handle interrupts gracefully
    let running = Arc::new(atomic::AtomicBool::new(true));
    let r = running.clone();
    ctrlc::set_handler(move || {
        r.store(false, atomic::Ordering::Relaxed);
    })
    .expect("setting Ctrl-C handler");

    println!("-> ENTRY");
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
    println!("-> INSTANCE");
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
    let app_info = vk::ApplicationInfo::builder()
        .application_version(0)
        .engine_version(0)
        .api_version(vk_target_version);

    println!("-> VK_INSTANCE");
    let gfx_instance = if true {
        // APPROACH NO. 2 -- most probably need to use this as per openxr docs
        let vk_instance = unsafe {
            xr_instance.create_vulkan_instance(
                system,
                std::mem::transmute(vk_entry.static_fn().get_instance_proc_addr),
                &vk::InstanceCreateInfo::builder().application_info(&app_info) as *const _
                    as *const _,
            )
        }
        .expect("XR error creating Vulkan instance")
        .map_err(vk::Result::from_raw)
        .expect("Vulkan error creating Vulkan instance");

        let vk_instance = unsafe {
            ash::Instance::load(
                vk_entry.static_fn(),
                vk::Instance::from_raw(vk_instance as _),
            )
        };

        let gfx_instance = back::Instance::from_raw(vk_instance).unwrap();
        gfx_instance
    } else {
        // APPROACH NO. 1
        back::Instance::create("openxrs example", 1).unwrap()
    };

    let gfx_adapters = gfx_instance.enumerate_adapters();
    let FIXME_gfx_adapter = gfx_adapters.first().unwrap();
    let gfx_physical_device = &FIXME_gfx_adapter.physical_device;

    println!("-> VK_PHYSICAL_DEVICE {:p}", unsafe {
        gfx_instance.as_raw().as_ptr()
    },);

    //// FIXME <--- gfx_adapter instead of vk_physical_device !
    let vk_physical_device = vk::PhysicalDevice::from_raw(
        xr_instance
            .vulkan_graphics_device(system, unsafe { gfx_instance.as_raw().as_ptr() } as _)
            //.vulkan_graphics_device(system, vk_instance.handle().as_raw() as _)
            .unwrap() as _,
    );

    let gfx_device_properties = FIXME_gfx_adapter.physical_device.properties();
    //     let vk_device_properties = vk_instance.get_physical_device_properties(vk_physical_device);
    //     if vk_device_properties.api_version < vk_target_version {
    //         vk_instance.destroy_instance(None);
    //         panic!("Vulkan phyiscal device doesn't support version 1.1");
    //     }

    let gfx_queue_family = FIXME_gfx_adapter
        .queue_families
        .iter()
        .find(|info| info.queue_type().supports_graphics())
        .unwrap();

    //     let queue_family_index = vk_instance
    //         .get_physical_device_queue_family_properties(vk_physical_device)
    //         .into_iter()
    //         .enumerate()
    //         .find_map(|(queue_family_index, info)| {
    //             if info.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
    //                 Some(queue_family_index as u32)
    //             } else {
    //                 None
    //             }
    //         })
    //         .expect("Vulkan device has no graphics queue");

    let mut gfx_requested_features = gfx_hal::Features::empty();
    gfx_requested_features.insert(gfx_hal::Features::MULTI_VIEWPORTS); // <-- TODO: is this multiview?

    let mut gfx_gpu = unsafe {
        gfx_physical_device
            .open(&[(gfx_queue_family, &[1.0])], gfx_requested_features)
            .unwrap()
    };

    let gfx_device = &mut gfx_gpu.device;

    //     let vk_device = xr_instance
    //         .create_vulkan_device(
    //             system,
    //             std::mem::transmute(vk_entry.static_fn().get_instance_proc_addr),
    //             vk_physical_device.as_raw() as _,
    //             &vk::DeviceCreateInfo::builder()
    //                 .queue_create_infos(&[vk::DeviceQueueCreateInfo::builder()
    //                     .queue_family_index(queue_family_index)
    //                     .queue_priorities(&[1.0])
    //                     .build()])
    //                 .push_next(&mut vk::PhysicalDeviceVulkan11Features {
    //                     multiview: vk::TRUE,
    //                     ..Default::default()
    //                 }) as *const _ as *const _,
    //         )
    //         .expect("XR error creating Vulkan device")
    //         .map_err(vk::Result::from_raw)
    //         .expect("Vulkan error creating Vulkan device");
    //     let vk_device =
    //         ash::Device::load(vk_instance.fp_v1_0(), vk::Device::from_raw(vk_device as _));

    let (gfx_queue_id, gfx_queue) = gfx_gpu
        .queue_groups
        .iter_mut()
        .enumerate()
        .find_map(|(idx, qq)| {
            if qq.family == gfx_queue_family.id() {
                Some((idx, qq.queues.first_mut().unwrap()))
            } else {
                None
            }
        })
        .unwrap();
    // FIXME: get queue by queue family index
    //     let queue = vk_device.get_device_queue(queue_family_index, 0);

    let view_mask = !(!0 << VIEW_COUNT);

    println!("-> RENDER_PASS");
    let gfx_render_pass = unsafe {
        gfx_device.create_render_pass(
            iter::once(pass::Attachment {
                format: None, //Some(COLOR_FORMAT),
                samples: 1,
                ops: pass::AttachmentOps::new(
                    pass::AttachmentLoadOp::Clear,
                    pass::AttachmentStoreOp::Store,
                ),
                stencil_ops: pass::AttachmentOps::DONT_CARE,
                layouts: image::Layout::Undefined..image::Layout::ColorAttachmentOptimal,
            }),
            iter::once(pass::SubpassDesc {
                // FIXME: .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS) missing
                colors: &[(0, image::Layout::ColorAttachmentOptimal)],
                depth_stencil: None,
                inputs: &[],
                resolves: &[],
                preserves: &[],
            }), // FIXME
            iter::empty(), // FIXME
                           /*
                           SubpassDependency {
                               passes: None..None,
                               stages: ,
                               accesses: image::Access::empty()..image::Access::empty(),
                               flags: Dependencies::empty(), // FIXME
                           } */
                           //FIXME: push multiview
        )
    }
    .unwrap();

    //     let render_pass = vk_device
    //         .create_render_pass(
    //             &vk::RenderPassCreateInfo::builder()
    //                 .attachments(&[vk::AttachmentDescription {
    //                     format: COLOR_FORMAT,
    //                     samples: vk::SampleCountFlags::TYPE_1,
    //                     load_op: vk::AttachmentLoadOp::CLEAR,
    //                     store_op: vk::AttachmentStoreOp::STORE,
    //                     initial_layout: vk::ImageLayout::UNDEFINED,
    //                     final_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
    //                     ..Default::default()
    //                 }])
    //                 .subpasses(&[vk::SubpassDescription::builder()
    //                     .color_attachments(&[vk::AttachmentReference {
    //                         attachment: 0,
    //                         layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
    //                     }])
    //                     .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
    //                     .build()])
    //                 .dependencies(&[vk::SubpassDependency {
    //                     src_subpass: vk::SUBPASS_EXTERNAL,
    //                     dst_subpass: 0,
    //                     src_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
    //                     dst_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
    //                     dst_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
    //                     ..Default::default()
    //                 }])
    //                 .push_next(
    //                     &mut vk::RenderPassMultiviewCreateInfo::builder()
    //                         .view_masks(&[view_mask])
    //                         .correlation_masks(&[view_mask]),
    //                 ),
    //             None,
    //         )
    //         .unwrap();

    let vert = read_spv(&mut Cursor::new(&include_bytes!("fullscreen.vert.spv")[..])).unwrap();
    let frag = read_spv(&mut Cursor::new(
        &include_bytes!("debug_pattern.frag.spv")[..],
    ))
    .unwrap();

    let gfx_vert = unsafe { gfx_device.create_shader_module(&vert) }.unwrap();
    //     let vert = vk_device
    //         .create_shader_module(&vk::ShaderModuleCreateInfo::builder().code(&vert), None)
    //         .unwrap();

    let gfx_frag = unsafe { gfx_device.create_shader_module(&frag) }.unwrap();
    //     let frag = vk_device
    //         .create_shader_module(&vk::ShaderModuleCreateInfo::builder().code(&frag), None)
    //         .unwrap();

    let gfx_pipeline_layout =
        unsafe { gfx_device.create_pipeline_layout(iter::empty(), iter::empty()) }.unwrap();

    //     let pipeline_layout = vk_device
    //         .create_pipeline_layout(
    //             &vk::PipelineLayoutCreateInfo::builder().set_layouts(&[]),
    //             None,
    //         )
    //         .unwrap();

    // FIXME: seems like this is part of GraphicsPipelineInfoBuf
    // let noop_stencil_state = back::conv::map_stencil_side(StencilFace::default());

    //     let noop_stencil_state = vk::StencilOpState {
    //         fail_op: vk::StencilOp::KEEP,
    //         pass_op: vk::StencilOp::KEEP,
    //         depth_fail_op: vk::StencilOp::KEEP,
    //         compare_op: vk::CompareOp::ALWAYS,
    //         compare_mask: 0,
    //         write_mask: 0,
    //         reference: 0,
    //     };

    println!("-> PIPELINE");
    let gfx_pipeline = unsafe {
        gfx_device.create_graphics_pipeline(
            // FIXME: viewport_state ? rasterization_state ? multisample_state? depth_stencil?
            &pso::GraphicsPipelineDesc::new(
                pso::PrimitiveAssemblerDesc::Vertex {
                    buffers: &vec![],
                    attributes: &vec![],
                    input_assembler: pso::InputAssemblerDesc {
                        primitive: pso::Primitive::TriangleList,
                        with_adjacency: false,
                        restart_index: None,
                    },
                    vertex: pso::EntryPoint {
                        entry: "main",
                        module: &gfx_vert,
                        specialization: Default::default(),
                    },
                    geometry: None,
                    tessellation: None,
                },
                pso::Rasterizer::FILL,
                Some(pso::EntryPoint {
                    entry: "main",
                    module: &gfx_frag,
                    specialization: Default::default(),
                }),
                &gfx_pipeline_layout,
                pass::Subpass {
                    index: 0,
                    main_pass: &gfx_render_pass,
                },
            ),
            None,
        )
    }
    .unwrap();

    //     let pipeline = vk_device
    //         .create_graphics_pipelines(
    //             vk::PipelineCache::null(),
    //             &[vk::GraphicsPipelineCreateInfo::builder()
    //                 .stages(&[
    //                     vk::PipelineShaderStageCreateInfo {
    //                         stage: vk::ShaderStageFlags::VERTEX,
    //                         module: vert,
    //                         p_name: b"main\0".as_ptr() as _,
    //                         ..Default::default()
    //                     },
    //                     vk::PipelineShaderStageCreateInfo {
    //                         stage: vk::ShaderStageFlags::FRAGMENT,
    //                         module: frag,
    //                         p_name: b"main\0".as_ptr() as _,
    //                         ..Default::default()
    //                     },
    //                 ])
    //                 .vertex_input_state(&vk::PipelineVertexInputStateCreateInfo::default())
    //                 .input_assembly_state(
    //                     &vk::PipelineInputAssemblyStateCreateInfo::builder()
    //                         .topology(vk::PrimitiveTopology::TRIANGLE_LIST),
    //                 )
    //                 .viewport_state(
    //                     &vk::PipelineViewportStateCreateInfo::builder()
    //                         .scissor_count(1)
    //                         .viewport_count(1),
    //                 )
    //                 .rasterization_state(
    //                     &vk::PipelineRasterizationStateCreateInfo::builder()
    //                         .cull_mode(vk::CullModeFlags::NONE)
    //                         .polygon_mode(vk::PolygonMode::FILL)
    //                         .line_width(1.0),
    //                 )
    //                 .multisample_state(
    //                     &vk::PipelineMultisampleStateCreateInfo::builder()
    //                         .rasterization_samples(vk::SampleCountFlags::TYPE_1),
    //                 )
    //                 .depth_stencil_state(
    //                     &vk::PipelineDepthStencilStateCreateInfo::builder()
    //                         .depth_test_enable(false)
    //                         .depth_write_enable(false)
    //                         .front(noop_stencil_state)
    //                         .back(noop_stencil_state),
    //                 )
    //                 .color_blend_state(
    //                     &vk::PipelineColorBlendStateCreateInfo::builder().attachments(&[
    //                         vk::PipelineColorBlendAttachmentState {
    //                             blend_enable: vk::TRUE,
    //                             src_color_blend_factor: vk::BlendFactor::ONE,
    //                             dst_color_blend_factor: vk::BlendFactor::ZERO,
    //                             color_blend_op: vk::BlendOp::ADD,
    //                             color_write_mask: vk::ColorComponentFlags::R
    //                                 | vk::ColorComponentFlags::G
    //                                 | vk::ColorComponentFlags::B,
    //                             ..Default::default()
    //                         },
    //                     ]),
    //                 )
    //                 .dynamic_state(
    //                     &vk::PipelineDynamicStateCreateInfo::builder().dynamic_states(&[
    //                         vk::DynamicState::VIEWPORT,
    //                         vk::DynamicState::SCISSOR,
    //                     ]),
    //                 )
    //                 .layout(pipeline_layout)
    //                 .render_pass(render_pass)
    //                 .subpass(0)
    //                 .build()],
    //             None,
    //         )
    //         .unwrap()[0];

    unsafe { gfx_device.destroy_shader_module(gfx_vert) };
    //     vk_device.destroy_shader_module(vert, None);

    unsafe { gfx_device.destroy_shader_module(gfx_frag) };
    //     vk_device.destroy_shader_module(frag, None);

    println!("-> SESSION");
    let (gfx_session, mut xr_frame_wait, mut xr_frame_stream) = unsafe {
        xr_instance.create_session::<xr::Gfx<gfx_backend_vulkan::Backend>>(
            system,
            &xr::gfx::SessionCreateInfo {
                instance: gfx_instance,
                physical_device: vk_physical_device.as_raw() as _,
                device: gfx_device.as_raw(),
                queue_family: gfx_queue_family.id(),
                queue_id: gfx_queue_id as u32,
            },
        )
    }
    .unwrap();

    //     // A session represents this application's desire to display things! This is where we hook
    //     // up our graphics API. This does not start the session; for that, you'll need a call to
    //     // Session::begin, which we do in 'main_loop below.
    //     let (session, mut frame_wait, mut frame_stream) = xr_instance
    //         .create_session::<xr::Vulkan>(
    //             system,
    //             &xr::vulkan::SessionCreateInfo {
    //                 instance: vk_instance.handle().as_raw() as _,
    //                 physical_device: vk_physical_device.as_raw() as _,
    //                 device: vk_device.handle().as_raw() as _,
    //                 queue_family_index,
    //                 queue_index: 0,
    //             },
    //         )
    //         .unwrap();

    // Create an action set to encapsulate our actions
    let action_set = xr_instance
        .create_action_set("input", "input pose information", 0)
        .unwrap();

    let right_action = action_set
        .create_action::<xr::Posef>("right_hand", "Right Hand Controller", &[])
        .unwrap();
    let left_action = action_set
        .create_action::<xr::Posef>("left_hand", "Left Hand Controller", &[])
        .unwrap();

    // Bind our actions to input devices using the given profile
    // If you want to access inputs specific to a particular device you may specify a different
    // interaction profile
    xr_instance
        .suggest_interaction_profile_bindings(
            xr_instance
                .string_to_path("/interaction_profiles/khr/simple_controller")
                .unwrap(),
            &[
                xr::Binding::new(
                    &right_action,
                    xr_instance
                        .string_to_path("/user/hand/right/input/grip/pose")
                        .unwrap(),
                ),
                xr::Binding::new(
                    &left_action,
                    xr_instance
                        .string_to_path("/user/hand/left/input/grip/pose")
                        .unwrap(),
                ),
            ],
        )
        .unwrap();

    // Attach the action set to the session
    gfx_session.attach_action_sets(&[&action_set]).unwrap();

    // Create an action space for each device we want to locate
    let right_space = right_action
        .create_space(gfx_session.clone(), xr::Path::NULL, xr::Posef::IDENTITY)
        .unwrap();
    let left_space = left_action
        .create_space(gfx_session.clone(), xr::Path::NULL, xr::Posef::IDENTITY)
        .unwrap();

    // OpenXR uses a couple different types of reference frames for positioning content; we need
    // to choose one for displaying our content! STAGE would be relative to the center of your
    // guardian system's bounds, and LOCAL would be relative to your device's starting location.
    let stage = gfx_session
        .create_reference_space(xr::ReferenceSpaceType::STAGE, xr::Posef::IDENTITY)
        .unwrap();

    //     let cmd_pool = vk_device
    //         .create_command_pool(
    //             &vk::CommandPoolCreateInfo::builder()
    //                 .queue_family_index(queue_family_index)
    //                 .flags(
    //                     vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER
    //                         | vk::CommandPoolCreateFlags::TRANSIENT,
    //                 ),
    //             None,
    //         )
    //         .unwrap();

    println!("-> COMMAND_POOL");
    let mut cmd_pool = unsafe {
        gfx_device.create_command_pool(gfx_queue_family.id(), pool::CommandPoolCreateFlags::empty())
    }
    .unwrap();

    let mut cmds = Vec::new();
    unsafe { cmd_pool.allocate(PIPELINE_DEPTH as usize, command::Level::Primary, &mut cmds) };

    //     let cmds = vk_device
    //         .allocate_command_buffers(
    //             &vk::CommandBufferAllocateInfo::builder()
    //                 .command_pool(cmd_pool)
    //                 .command_buffer_count(PIPELINE_DEPTH),
    //         )
    //         .unwrap();
    //
    println!("-> FENCES");
    let mut fences = (0..PIPELINE_DEPTH)
        .map(|_| gfx_device.create_fence(true).unwrap())
        .collect::<Vec<_>>();

    //     let fences = (0..PIPELINE_DEPTH)
    //         .map(|_| {
    //             vk_device
    //                 .create_fence(
    //                     &vk::FenceCreateInfo::builder().flags(vk::FenceCreateFlags::SIGNALED),
    //                     None,
    //                 )
    //                 .unwrap()
    //         })
    //         .collect::<Vec<_>>();

    // Main loop
    let mut swapchain = None;
    let mut event_storage = xr::EventDataBuffer::new();
    let mut session_running = false;
    // Index of the current frame, wrapped by PIPELINE_DEPTH. Not to be confused with the
    // swapchain image index.
    let mut frame = 0;
    println!("-> LOOP");
    'main_loop: loop {
        if !running.load(atomic::Ordering::Relaxed) {
            println!("requesting exit");
            // The OpenXR runtime may want to perform a smooth transition between scenes, so we
            // can't necessarily exit instantly. Instead, we must notify the runtime of our
            // intent and wait for it to tell us when we're actually done.
            match gfx_session.request_exit() {
                Ok(()) => {}
                Err(xr::sys::Result::ERROR_SESSION_NOT_RUNNING) => break,
                Err(e) => panic!("{}", e),
            }
        }

        while let Some(event) = xr_instance.poll_event(&mut event_storage).unwrap() {
            use xr::Event::*;
            match event {
                SessionStateChanged(e) => {
                    // Session state change is where we can begin and end sessions, as well as
                    // find quit messages!
                    println!("entered state {:?}", e.state());
                    match e.state() {
                        xr::SessionState::READY => {
                            gfx_session.begin(VIEW_TYPE).unwrap();
                            session_running = true;
                        }
                        xr::SessionState::STOPPING => {
                            gfx_session.end().unwrap();
                            session_running = false;
                        }
                        xr::SessionState::EXITING | xr::SessionState::LOSS_PENDING => {
                            break 'main_loop;
                        }
                        _ => {}
                    }
                }
                InstanceLossPending(_) => {
                    break 'main_loop;
                }
                EventsLost(e) => {
                    println!("lost {} events", e.lost_event_count());
                }
                _ => {}
            }
        }

        if !session_running {
            // Don't grind up the CPU
            std::thread::sleep(Duration::from_millis(100));
            continue;
        }

        // Block until the previous frame is finished displaying, and is ready for another one.
        // Also returns a prediction of when the next frame will be displayed, for use with
        // predicting locations of controllers, viewpoints, etc.
        let xr_frame_state = xr_frame_wait.wait().unwrap();
        // Must be called before any rendering is done!
        xr_frame_stream.begin().unwrap();

        if !xr_frame_state.should_render {
            xr_frame_stream
                .end(
                    xr_frame_state.predicted_display_time,
                    environment_blend_mode,
                    &[],
                )
                .unwrap();
            continue;
        }

        let swapchain = swapchain.get_or_insert_with(|| {
            // Now we need to find all the viewpoints we need to take care of! This is a
            // property of the view configuration type; in this example we use PRIMARY_STEREO,
            // so we should have 2 viewpoints.
            //
            // Because we are using multiview in this example, we require that all view
            // dimensions are identical.
            let views = xr_instance
                .enumerate_view_configuration_views(system, VIEW_TYPE)
                .unwrap();
            assert_eq!(views.len(), VIEW_COUNT as usize);
            assert_eq!(views[0], views[1]);

            // Create a swapchain for the viewpoints! A swapchain is a set of texture buffers
            // used for displaying to screen, typically this is a backbuffer and a front buffer,
            // one for rendering data to, and one for displaying on-screen.
            let resolution = vk::Extent2D {
                width: views[0].recommended_image_rect_width,
                height: views[0].recommended_image_rect_height,
            };

            let format = vk::Format::from_raw(COLOR_FORMAT as i32);
            assert_eq!(format, vk::Format::B8G8R8A8_SRGB);

            let handle = gfx_session
                .create_swapchain(&xr::SwapchainCreateInfo {
                    create_flags: xr::SwapchainCreateFlags::EMPTY,
                    usage_flags: xr::SwapchainUsageFlags::COLOR_ATTACHMENT
                        | xr::SwapchainUsageFlags::SAMPLED,
                    format: format.as_raw() as _,
                    // The Vulkan graphics pipeline we create is not set up for multisampling,
                    // so we hardcode this to 1. If we used a proper multisampling setup, we
                    // could set this to `views[0].recommended_swapchain_sample_count`.
                    sample_count: 1,
                    width: resolution.width,
                    height: resolution.height,
                    face_count: 1,
                    array_size: VIEW_COUNT,
                    mip_count: 1,
                })
                .unwrap();

            // We'll want to track our own information about the swapchain, so we can draw stuff
            // onto it! We'll also create a buffer for each generated texture here as well.
            let images = handle.enumerate_images().unwrap();
            Swapchain {
                handle,
                resolution,
                buffers: images
                    .into_iter()
                    .map(|color_image| {
                        //gfx_device.create_image(kind, mip_levels, format, tiling, usage, sparse, view_caps)
                        let image = back::Image::from_raw(
                            vk::Image::from_raw(color_image).as_raw() as *const c_void,
                            image::Kind::D2(1, 1, 1, 0),
                            image::ViewCapabilities::default(),
                        );

                        let image_view = unsafe {
                            gfx_device.create_image_view(
                                &image,
                                image::ViewKind::D2Array,
                                COLOR_FORMAT,
                                format::Swizzle::default(),
                                image::Usage::empty(),
                                image::SubresourceRange {
                                    aspects: format::Aspects::COLOR,
                                    level_start: 0,
                                    level_count: Some(1),
                                    layer_start: 0,
                                    layer_count: Some(VIEW_COUNT as u16),
                                },
                            )
                        }
                        .unwrap();

                        /*
                        let color = vk_device
                            .create_image_view(
                                &vk::ImageViewCreateInfo::builder()
                                    .image(color_image)
                                    .view_type(vk::ImageViewType::TYPE_2D_ARRAY)
                                    .format(COLOR_FORMAT)
                                    .subresource_range(vk::ImageSubresourceRange {
                                        aspect_mask: vk::ImageAspectFlags::COLOR,
                                        base_mip_level: 0,
                                        level_count: 1,
                                        base_array_layer: 0,
                                        layer_count: VIEW_COUNT,
                                    }),
                                None,
                            )
                            .unwrap();
                            */

                        let framebuffer = unsafe {
                            gfx_device.create_framebuffer(
                                &gfx_render_pass,
                                iter::once(image::FramebufferAttachment {
                                    usage: image::Usage::empty(),
                                    view_caps: image::ViewCapabilities::empty(),
                                    format: COLOR_FORMAT,
                                }),
                                image::Extent {
                                    width: resolution.width,
                                    height: resolution.height,
                                    depth: 1,
                                },
                            )
                        }
                        .unwrap();

                        /*
                                               let framebuffer = vk_device
                                                   .create_framebuffer(
                                                       &vk::FramebufferCreateInfo::builder()
                                                           .render_pass(render_pass)
                                                           .width(resolution.width)
                                                           .height(resolution.height)
                                                           .attachments(&[color])
                                                           .layers(1), // Multiview handles addressing multiple layers
                                                       None,
                                                   )
                                                   .unwrap();
                        */

                        Framebuffer {
                            framebuffer,
                            image,
                            image_view,
                        }
                    })
                    .collect(),
            }
        });

        // We need to ask which swapchain image to use for rendering! Which one will we get?
        // Who knows! It's up to the runtime to decide.
        let image_index = swapchain.handle.acquire_image().unwrap();
        println!("IMAGE_IDX: {}", image_index);

        // Wait until the image is available to render to. The compositor could still be
        // reading from it.
        swapchain.handle.wait_image(xr::Duration::INFINITE).unwrap();

        unsafe {
            gfx_device.wait_for_fences(iter::once(&fences[frame]), device::WaitFor::All, u64::MAX)
        }
        .unwrap();

        //         // Ensure the last use of this frame's resources is 100% done
        //         vk_device
        //             .wait_for_fences(&[fences[frame]], true, u64::MAX)
        //             .unwrap();

        unsafe { gfx_device.reset_fence(&mut fences[frame]) }.unwrap();
        //         vk_device.reset_fences(&[fences[frame]]).unwrap();

        let cmd = &mut cmds[frame];

        unsafe { cmd.begin_primary(command::CommandBufferFlags::ONE_TIME_SUBMIT) };
        //         let cmd = cmds[frame];
        //         vk_device
        //             .begin_command_buffer(
        //                 cmd,
        //                 &vk::CommandBufferBeginInfo::builder()
        //                     .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
        //             )
        //             .unwrap();

        let buffer = &swapchain.buffers[image_index as usize];
        unsafe {
            cmd.begin_render_pass(
                &gfx_render_pass,
                &buffer.framebuffer,
                pso::Rect {
                    x: 0,
                    y: 0,
                    w: swapchain.resolution.width as i16,
                    h: swapchain.resolution.height as i16,
                },
                iter::once(command::RenderAttachmentInfo {
                    image_view: &buffer.image_view,
                    clear_value: command::ClearValue::default(),
                }),
                command::SubpassContents::Inline,
            )
        };
        //         vk_device.cmd_begin_render_pass(
        //             cmd,
        //             &vk::RenderPassBeginInfo::builder()
        //                 .render_pass(render_pass)
        //                 .framebuffer(swapchain.buffers[image_index as usize].framebuffer)
        //                 .render_area(vk::Rect2D {
        //                     offset: vk::Offset2D::default(),
        //                     extent: swapchain.resolution,
        //                 })
        //                 .clear_values(&[vk::ClearValue {
        //                     color: vk::ClearColorValue {
        //                         float32: [0.0, 0.0, 0.0, 1.0],
        //                     },
        //                 }]),
        //             vk::SubpassContents::INLINE,
        //         );

        let viewport = pso::Viewport {
            rect: pso::Rect {
                x: 0,
                y: 0,
                w: swapchain.resolution.width as i16,
                h: swapchain.resolution.height as i16,
            },
            depth: 0.0..1.0,
        };
        //         let viewports = [vk::Viewport {
        //             x: 0.0,
        //             y: 0.0,
        //             width: swapchain.resolution.width as f32,
        //             height: swapchain.resolution.height as f32,
        //             min_depth: 0.0,
        //             max_depth: 1.0,
        //         }];
        let scissors = pso::Rect {
            x: 0,
            y: 0,
            w: swapchain.resolution.width as i16,
            h: swapchain.resolution.height as i16,
        };

        //         let scissors = [vk::Rect2D {
        //             offset: vk::Offset2D { x: 0, y: 0 },
        //             extent: swapchain.resolution,
        //         }];

        unsafe { cmd.set_viewports(0, iter::once(viewport)) };
        unsafe { cmd.set_scissors(0, iter::once(scissors)) };
        //         vk_device.cmd_set_viewport(cmd, 0, &viewports);
        //         vk_device.cmd_set_scissor(cmd, 0, &scissors);

        //         // Draw the scene. Multiview means we only need to do this once, and the GPU will
        //         // automatically broadcast operations to all views. Shaders can use `gl_ViewIndex` to
        //         // e.g. select the correct view matrix.
        unsafe { cmd.bind_graphics_pipeline(&gfx_pipeline) };
        //         vk_device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline);
        unsafe { cmd.draw(0..3, 0..1) };
        //         vk_device.cmd_draw(cmd, 3, 1, 0, 0);

        unsafe { cmd.end_render_pass() };
        //         vk_device.cmd_end_render_pass(cmd);

        unsafe { cmd.finish() };
        //         vk_device.end_command_buffer(cmd).unwrap();

        gfx_session.sync_actions(&[(&action_set).into()]).unwrap();

        // Find where our controllers are located in the Stage space
        let right_location = right_space
            .locate(&stage, xr_frame_state.predicted_display_time)
            .unwrap();

        let left_location = left_space
            .locate(&stage, xr_frame_state.predicted_display_time)
            .unwrap();

        if left_action.is_active(&gfx_session, xr::Path::NULL).unwrap() {
            print!(
                "Left Hand: ({:0<12},{:0<12},{:0<12}), ",
                left_location.pose.position.x,
                left_location.pose.position.y,
                left_location.pose.position.z
            );
        }

        if right_action
            .is_active(&gfx_session, xr::Path::NULL)
            .unwrap()
        {
            print!(
                "Right Hand: ({:0<12},{:0<12},{:0<12})",
                right_location.pose.position.x,
                right_location.pose.position.y,
                right_location.pose.position.z
            );
        }
        // println!();

        // Fetch the view transforms. To minimize latency, we intentionally do this *after*
        // recording commands to render the scene, i.e. at the last possible moment before
        // rendering begins in earnest on the GPU. Uniforms dependent on this data can be sent
        // to the GPU just-in-time by writing them to per-frame host-visible memory which the
        // GPU will only read once the command buffer is submitted.
        let (_, views) = gfx_session
            .locate_views(VIEW_TYPE, xr_frame_state.predicted_display_time, &stage)
            .unwrap();

        unsafe {
            gfx_queue.submit(
                iter::once(&cmds[frame]),
                iter::empty(),
                iter::empty(),
                Some(&mut fences[frame]),
            )
        };
        //         // Submit commands to the GPU, then tell OpenXR we're done with our part.
        //         vk_device
        //             .queue_submit(
        //                 queue,
        //                 &[vk::SubmitInfo::builder().command_buffers(&[cmd]).build()],
        //                 fences[frame],
        //             )
        //             .unwrap();
        swapchain.handle.release_image().unwrap();

        // Tell OpenXR what to present for this frame
        let rect = xr::Rect2Di {
            offset: xr::Offset2Di { x: 0, y: 0 },
            extent: xr::Extent2Di {
                width: swapchain.resolution.width as _,
                height: swapchain.resolution.height as _,
            },
        };

        xr_frame_stream
            .end(
                xr_frame_state.predicted_display_time,
                environment_blend_mode,
                &[
                    &xr::CompositionLayerProjection::new().space(&stage).views(&[
                        xr::CompositionLayerProjectionView::new()
                            .pose(views[0].pose)
                            .fov(views[0].fov)
                            .sub_image(
                                xr::SwapchainSubImage::new()
                                    .swapchain(&swapchain.handle)
                                    .image_array_index(0)
                                    .image_rect(rect),
                            ),
                        xr::CompositionLayerProjectionView::new()
                            .pose(views[1].pose)
                            .fov(views[1].fov)
                            .sub_image(
                                xr::SwapchainSubImage::new()
                                    .swapchain(&swapchain.handle)
                                    .image_array_index(1)
                                    .image_rect(rect),
                            ),
                    ]),
                ],
            )
            .unwrap();

        frame = (frame + 1) % PIPELINE_DEPTH as usize;
    }

    // OpenXR MUST be allowed to clean up before we destroy Vulkan resources it could touch, so
    // first we must drop all its handles.
    drop((
        gfx_session,
        xr_frame_wait,
        xr_frame_stream,
        stage,
        action_set,
        left_space,
        right_space,
        left_action,
        right_action,
    ));

    // Ensure all in-flight frames are finished before destroying resources they might use
    unsafe { gfx_device.wait_for_fences(fences.iter(), device::WaitFor::All, !0) }.unwrap();

    // FIXME: are these destructors really needed?
    for fence in fences {
        unsafe { gfx_device.destroy_fence(fence) };
    }

    if let Some(swapchain) = swapchain {
        for buffer in swapchain.buffers {
            unsafe { gfx_device.destroy_framebuffer(buffer.framebuffer) };
            unsafe { gfx_device.destroy_image_view(buffer.image_view) };
        }
    }

    unsafe {
        // FIXME
        //     vk_device.destroy_pipeline(pipeline, None);
        gfx_device.destroy_pipeline_layout(gfx_pipeline_layout);
        gfx_device.destroy_command_pool(cmd_pool);
        gfx_device.destroy_render_pass(gfx_render_pass);
    }
    //     vk_device.destroy_device(None);
    //     vk_instance.destroy_instance(None);
    // }

    println!("exiting cleanly");
}

// FIXME ok?
pub const COLOR_FORMAT: format::Format = format::Format::Bgra8Srgb;
// pub const COLOR_FORMAT: vk::Format = vk::Format::B8G8R8A8_SRGB;

pub const VIEW_COUNT: u32 = 2;
const VIEW_TYPE: xr::ViewConfigurationType = xr::ViewConfigurationType::PRIMARY_STEREO;

struct Swapchain<B: gfx_hal::Backend> {
    handle: xr::Swapchain<xr::Gfx<B>>,
    buffers: Vec<Framebuffer<B>>,
    resolution: vk::Extent2D,
}

struct Framebuffer<B: gfx_hal::Backend> {
    framebuffer: B::Framebuffer,
    image: B::Image,
    image_view: B::ImageView,
}

/// Maximum number of frames in flight
const PIPELINE_DEPTH: u32 = 2;
