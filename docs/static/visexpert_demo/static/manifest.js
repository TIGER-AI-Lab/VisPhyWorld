window.VISEXPERT_DEMO_MANIFEST = {
  "generated_at": "local",
  "root": "webdemo",
  "models": [
    {
      "key": "gpt-5",
      "label": "GPT-5"
    },
    {
      "key": "claude-sonnet-4-5",
      "label": "Claude Sonnet 4.5"
    },
    {
      "key": "gemini-3-pro",
      "label": "Gemini 3 Pro"
    },
    {
      "key": "gpt-4.1",
      "label": "GPT-4.1"
    },
    {
      "key": "qwen3-vl-plus",
      "label": "Qwen3-VL-Plus"
    }
  ],
  "samples": [
    {
      "id": "task10063_000",
      "label": "Case 1 (2D)",
      "prompt": "The scene is set against a plain white background with no visible ground plane, platforms, or mechanical supports. There are no walls, ramps, or other static obstacles; all visible elements appear to be floating in open space. The layout is vertically oriented, with objects roughly aligned along a central vertical axis, and ample empty space around them.\n\nThere are three circular objects and two thin vertical/diagonal bars. The largest dynamic object is a red circle centered slightly below the midpoint of the image, located on the central vertical axis; it is likely free to move under gravity and may fall straight down. Above it are two smaller grey circles: one slightly to the left of the axis and lower, the other higher and slightly to the right; both appear initially at rest and could also move vertically if affected by gravity. Near the bottom, there are two long, slender grey bars: the left bar is slanted, forming an “/” shape, while the right bar is vertical; both are positioned below the red circle and may act as potential contact surfaces if anything falls onto them.\n\nThe critical interactions will be governed primarily by gravity causing the circular objects to accelerate downward. The red circle, starting centrally above the bars, will likely descend first and may collide with either the slanted left bar, the vertical right bar, or slip between/along them depending on alignment. Contact with the slanted bar could induce sliding along its length and a horizontal deflection, while a direct hit on the vertical bar would cause a more elastic bounce or redirection. The smaller grey circles, if also free, may follow similar vertical paths and could collide with the red circle or each other, altering their trajectories via momentum exchange before eventually interacting with the lower bars or continuing downward past them.",
      "gt_frames": {
        "first": "assets/samples/task10063_000/frame_01.png",
        "tenth": "assets/samples/task10063_000/frame_10.png"
      },
      "gt_video": "assets/samples/task10063_000/gt.mp4",
      "models": [
        {
          "key": "gpt-5",
          "label": "GPT-5",
          "video": "assets/samples/task10063_000/models/gpt-5.mp4"
        },
        {
          "key": "claude-sonnet-4-5",
          "label": "Claude Sonnet 4.5",
          "video": "assets/samples/task10063_000/models/claude-sonnet-4-5.mp4"
        },
        {
          "key": "gemini-3-pro",
          "label": "Gemini 3 Pro",
          "video": "assets/samples/task10063_000/models/gemini-3-pro.mp4"
        },
        {
          "key": "gpt-4.1",
          "label": "GPT-4.1",
          "video": "assets/samples/task10063_000/models/gpt-4.1.mp4"
        },
        {
          "key": "qwen3-vl-plus",
          "label": "Qwen3-VL-Plus",
          "video": "assets/samples/task10063_000/models/qwen3-vl-plus.mp4"
        }
      ]
    },
    {
      "id": "06_tmpl_06",
      "label": "Case 2 (3D)",
      "prompt": "The scene is set on a large, flat, light-gray platform that serves as the ground. On the right side of the platform there is a U-shaped, translucent light-gray container or open box fixed to the ground, with two parallel side walls and a back wall, open toward the left. Near the center of the platform lies a long, thin, black plank or wedge, slightly raised and oriented diagonally from lower-left to upper-right, acting as a potential ramp or deflector. All these structures appear static and anchored to the ground.\n\nThere is one clearly dynamic object: a small green sphere located to the left of the black plank. Between frame 1 and frame 10, the sphere’s position shifts slightly to the right and marginally upward in the image plane, indicating a gentle motion toward the black plank, roughly along a line from left to right. The motion appears slow, with no visible deformation, consistent with a rigid ball rolling or sliding across the flat surface under some initial impulse and gravity.\n\nGiven the frames, the main anticipated interaction is between the moving green sphere and the central black plank. As the ball travels rightward, it is likely to contact the inclined plank, where collision and friction may redirect its motion, possibly deflecting it toward the U-shaped container on the right. Gravity keeps the ball on the surface, while contact forces with the ground enable rolling or sliding. If the ball’s path and speed are sufficient, it may strike the plank first, change trajectory, and then either enter or miss the open side of the container, establishing a cause-effect chain of initial motion → impact with the ramp → redirected trajectory toward the container.",
      "gt_frames": {
        "first": "assets/samples/06_tmpl_06/frame_01.png",
        "tenth": "assets/samples/06_tmpl_06/frame_10.png"
      },
      "gt_video": "assets/samples/06_tmpl_06/gt.mp4",
      "models": [
        {
          "key": "gpt-5",
          "label": "GPT-5",
          "video": "assets/samples/06_tmpl_06/models/gpt-5.mp4"
        },
        {
          "key": "claude-sonnet-4-5",
          "label": "Claude Sonnet 4.5",
          "video": "assets/samples/06_tmpl_06/models/claude-sonnet-4-5.mp4"
        },
        {
          "key": "gemini-3-pro",
          "label": "Gemini 3 Pro",
          "video": "assets/samples/06_tmpl_06/models/gemini-3-pro.mp4"
        },
        {
          "key": "gpt-4.1",
          "label": "GPT-4.1",
          "video": "assets/samples/06_tmpl_06/models/gpt-4.1.mp4"
        },
        {
          "key": "qwen3-vl-plus",
          "label": "Qwen3-VL-Plus",
          "video": "assets/samples/06_tmpl_06/models/qwen3-vl-plus.mp4"
        }
      ]
    },
    {
      "id": "14_tmpl_14",
      "label": "Case 3 (3D)",
      "prompt": "The scene consists of a large, flat, grey horizontal ground plane occupying most of the view. Centered on this plane is a U‑shaped static structure made of three rectangular grey walls: a flat base aligned with the ground and two vertical side walls at its left and right edges, open toward the viewer. There are no other fixed obstacles or supports visible besides this low, open container-like structure and the ground plane.\n\nThere are two dynamic objects: a red sphere and a blue sphere. Both spheres start above the U‑shaped structure, roughly centered over the base region. The red sphere is slightly to the left side while the blue sphere is slightly to the right and a bit higher. Between the first and tenth frame, both balls move downward under gravity toward the opening of the U‑shape; their trajectories appear nearly vertical with a slight horizontal offset preserved (red left, blue right). No significant lateral motion or rotation is evident from the frames, only vertical descent.\n\nThe critical interactions will be governed by gravity pulling the spheres down toward the U‑shaped structure. As they fall, each sphere is likely to strike the corresponding side or the inner base of the U, depending on its exact lateral alignment. If aligned over the base, they will first collide with the U’s floor, potentially bouncing slightly before settling. If slightly offset, one or both may contact a side wall first, causing a deflection and possible secondary collision with the base or opposite wall. All subsequent motion—bouncing, rolling, or resting—will follow from these collisions and the constraints imposed by the U‑shaped enclosure and friction with its surfaces.",
      "gt_frames": {
        "first": "assets/samples/14_tmpl_14/frame_01.png",
        "tenth": "assets/samples/14_tmpl_14/frame_10.png"
      },
      "gt_video": "assets/samples/14_tmpl_14/gt.mp4",
      "models": [
        {
          "key": "gpt-5",
          "label": "GPT-5",
          "video": "assets/samples/14_tmpl_14/models/gpt-5.mp4"
        },
        {
          "key": "claude-sonnet-4-5",
          "label": "Claude Sonnet 4.5",
          "video": "assets/samples/14_tmpl_14/models/claude-sonnet-4-5.mp4"
        },
        {
          "key": "gemini-3-pro",
          "label": "Gemini 3 Pro",
          "video": "assets/samples/14_tmpl_14/models/gemini-3-pro.mp4"
        },
        {
          "key": "gpt-4.1",
          "label": "GPT-4.1",
          "video": "assets/samples/14_tmpl_14/models/gpt-4.1.mp4"
        },
        {
          "key": "qwen3-vl-plus",
          "label": "Qwen3-VL-Plus",
          "video": "assets/samples/14_tmpl_14/models/qwen3-vl-plus.mp4"
        }
      ]
    },
    {
      "id": "task10055_011",
      "label": "Case 4 (2D)",
      "prompt": "The scene is set in a mostly empty vertical space with no visible ground plane, but there is a U‑shaped container near the bottom made of two light-gray vertical segments joined by a horizontal base, forming an open-topped receptacle. To the left of this container, slightly above its rim height, there is a single black diagonal bar, slanted upward from left to right, acting as an inclined surface. There are no other static walls or platforms; the background is plain.\n\nThere are two notable dynamic objects. Near the top center of the scene is a light-gray circular ball aligned vertically above the U-shaped container; directly below this ball, along the same vertical line, is a short light-gray vertical bar segment that may initially support or align the top ball’s motion. To the right of this vertical alignment, roughly midway up the scene and horizontally level with the black bar’s upper region, is a red circular ball that starts in midair with no immediate contact with other objects.\n\nUnder gravity, both balls are expected to fall downward. The top light-gray ball will move straight down; depending on precise alignment, it may pass by the gray vertical segment (if only visual) or slide along it before continuing toward the U-shaped container, eventually colliding with and settling inside it. The red ball will fall vertically until it reaches the level of the black diagonal bar; it is likely to strike this bar, bounce or roll along its downward-sloping direction from right to left, and then be redirected toward the interior of the U-shaped container. The critical interactions therefore occur in this order: gravitational fall of each ball, potential glancing interaction between the top gray ball and its vertical bar, impact of the red ball with the black inclined bar, sliding/rolling along the bar, and final capture of one or both balls inside the U-shaped container due to the confining walls and base.",
      "gt_frames": {
        "first": "assets/samples/task10055_011/frame_01.png",
        "tenth": "assets/samples/task10055_011/frame_10.png"
      },
      "gt_video": "assets/samples/task10055_011/gt.mp4",
      "models": [
        {
          "key": "gpt-5",
          "label": "GPT-5",
          "video": "assets/samples/task10055_011/models/gpt-5.mp4"
        },
        {
          "key": "claude-sonnet-4-5",
          "label": "Claude Sonnet 4.5",
          "video": "assets/samples/task10055_011/models/claude-sonnet-4-5.mp4"
        },
        {
          "key": "gemini-3-pro",
          "label": "Gemini 3 Pro",
          "video": "assets/samples/task10055_011/models/gemini-3-pro.mp4"
        },
        {
          "key": "gpt-4.1",
          "label": "GPT-4.1",
          "video": "assets/samples/task10055_011/models/gpt-4.1.mp4"
        },
        {
          "key": "qwen3-vl-plus",
          "label": "Qwen3-VL-Plus",
          "video": "assets/samples/task10055_011/models/qwen3-vl-plus.mp4"
        }
      ]
    },
    {
      "id": "task00000_000",
      "label": "Case 5 (2D)",
      "prompt": "The scene consists of a blank, white background with no visible ground line, platforms, walls, or other static supports. There are no ramps, pegs, or obstacles; the space appears open and unobstructed in both horizontal and vertical directions. The objects seem to float in this empty 2D field, implying that any motion will be governed mainly by gravity in the vertical direction and free translation horizontally.\n\nThere are three dynamic circular objects. The first is a large red circle located in the upper central region of the scene, clearly above the other two objects and roughly centered horizontally. The second is a smaller teal (light greenish-blue) circle positioned near the bottom-left portion of the frame. The third is a medium–large blue circle placed near the bottom-right portion of the frame, horizontally separated from the teal circle and vertically aligned at a similar lower height. None of the circles are initially touching each other.\n\nUnder gravity, the red circle is expected to move downward, potentially along a straight vertical trajectory since there are no visible horizontal forces or constraints. The teal and blue circles, being lower, may remain stationary until contacted by the descending red circle. The likely interaction sequence is that the red circle falls and first collides with either the teal or blue circle depending on precise alignment; the impact could then transfer momentum, causing the contacted lower circle to move sideways and possibly collide with the other lower circle, producing a secondary collision. The overall cause-effect chain is thus a gravity-driven fall of the red circle, followed by one or more elastic or inelastic collisions among the three circles in the otherwise empty space.",
      "gt_frames": {
        "first": "assets/samples/task00000_000/frame_01.png",
        "tenth": "assets/samples/task00000_000/frame_10.png"
      },
      "gt_video": "assets/samples/task00000_000/gt.mp4",
      "models": [
        {
          "key": "gpt-5",
          "label": "GPT-5",
          "video": "assets/samples/task00000_000/models/gpt-5.mp4"
        },
        {
          "key": "claude-sonnet-4-5",
          "label": "Claude Sonnet 4.5",
          "video": "assets/samples/task00000_000/models/claude-sonnet-4-5.mp4"
        },
        {
          "key": "gemini-3-pro",
          "label": "Gemini 3 Pro",
          "video": "assets/samples/task00000_000/models/gemini-3-pro.mp4"
        },
        {
          "key": "gpt-4.1",
          "label": "GPT-4.1",
          "video": "assets/samples/task00000_000/models/gpt-4.1.mp4"
        },
        {
          "key": "qwen3-vl-plus",
          "label": "Qwen3-VL-Plus",
          "video": "assets/samples/task00000_000/models/qwen3-vl-plus.mp4"
        }
      ]
    },
    {
      "id": "task10009_013",
      "label": "Case 6 (2D)",
      "prompt": "The scene is set against a plain white background with no visible ground plane or enclosing walls. There are no fixed platforms or supports explicitly drawn; instead, the reference frame shows open space with a few geometric elements that may act as obstacles or informal reference lines. A thick black diagonal segment appears on the left side, slanting upward from lower left toward upper right, suggesting a possible sloped barrier or ramp-like structure. Near the bottom center is a very thin, pale vertical line that could serve as a narrow post or divider. No other rigid static structures such as floors or containers are visible.\n\nThe dynamic objects consist of several circles. On the right side, there is a large peach-colored circle positioned roughly in the mid‑right region; it appears free in space and is likely a movable object that can translate and possibly collide with others. Near the center, slightly left of the frame’s midline, there are three small black circles aligned roughly along a diagonal from lower left to upper right; their spacing and alignment suggest they may be successive positions of a single moving ball or three distinct small balls that could move along that diagonal path. Near the top center, there is a larger, very pale gray circle, somewhat translucent, that is centered above the others; it is also likely dynamic and may move vertically downward under gravity.\n\nThe critical interactions will be governed by gravity pulling all circular objects downward and by their mutual collisions with each other and with the slanted black segment or thin vertical line. The upper pale gray circle may fall first, potentially colliding with one of the small black circles if their paths intersect, transferring momentum and altering their trajectories. The aligned small black circles suggest motion along the same diagonal direction as the slanted black bar; they might slide or bounce along that imaginary line, or ricochet off the nearby thick black segment if they reach it, causing changes in direction due to elastic or partially inelastic impacts. The large peach circle on the right could be struck by one or more of the smaller circles, resulting in a noticeable deflection or rolling motion. Overall, the sequence of events will likely involve falling, bouncing, and deflecting motions as gravity accelerates the circles downward, with collisions between circles and with linear obstacles determining the order and outcome of interactions.",
      "gt_frames": {
        "first": "assets/samples/task10009_013/frame_01.png",
        "tenth": "assets/samples/task10009_013/frame_10.png"
      },
      "gt_video": "assets/samples/task10009_013/gt.mp4",
      "models": [
        {
          "key": "gpt-5",
          "label": "GPT-5",
          "video": "assets/samples/task10009_013/models/gpt-5.mp4"
        },
        {
          "key": "claude-sonnet-4-5",
          "label": "Claude Sonnet 4.5",
          "video": "assets/samples/task10009_013/models/claude-sonnet-4-5.mp4"
        },
        {
          "key": "gemini-3-pro",
          "label": "Gemini 3 Pro",
          "video": "assets/samples/task10009_013/models/gemini-3-pro.mp4"
        },
        {
          "key": "gpt-4.1",
          "label": "GPT-4.1",
          "video": "assets/samples/task10009_013/models/gpt-4.1.mp4"
        },
        {
          "key": "qwen3-vl-plus",
          "label": "Qwen3-VL-Plus",
          "video": "assets/samples/task10009_013/models/qwen3-vl-plus.mp4"
        }
      ]
    },
    {
      "id": "task10021_014",
      "label": "Case 7 (2D)",
      "prompt": "The scene is set in an open white space without a visible ground plane. Near the bottom center, there is a U‑shaped gray container or basket composed of two vertical bars with a horizontal bar connecting their lower ends, forming an open top. Slightly above and centered over this container is open space where other objects can move and potentially fall into the container. No other static platforms or walls are visible.\n\nThere are several dynamic objects. Near the top center is a large red sphere attached on its lower-right side to a short gray rod that hangs downward; this suggests a pendulum-like element that could swing. Below and slightly left of this red sphere is a smaller gray sphere, currently positioned in midair. To the left of this gray sphere, there is a sequence of four small black spheres forming a curved, descending dotted path from upper left toward the gray sphere, indicating the past or projected trajectory of a moving object—likely representing the motion of the gray sphere or an identical one along a ballistic arc toward the central region and ultimately toward the container.\n\nThe critical interactions likely begin with the smaller gray sphere moving along the indicated curved path under gravity, perhaps having been released or deflected earlier. As it travels, it approaches the region beneath the red sphere and its rod; depending on exact alignment, it may either just pass by or collide with the rod or red sphere, potentially imparting a small impulse and causing the red sphere/rod assembly to rotate or swing. After this interaction (or simple free fall if no contact occurs), the gray sphere continues downward under gravity toward the U‑shaped container, where it is expected to either fall into and be captured by the container or possibly strike its rim and bounce, all governed by gravitational acceleration and elastic collision dynamics.",
      "gt_frames": {
        "first": "assets/samples/task10021_014/frame_01.png",
        "tenth": "assets/samples/task10021_014/frame_10.png"
      },
      "gt_video": "assets/samples/task10021_014/gt.mp4",
      "models": [
        {
          "key": "gpt-5",
          "label": "GPT-5",
          "video": "assets/samples/task10021_014/models/gpt-5.mp4"
        },
        {
          "key": "claude-sonnet-4-5",
          "label": "Claude Sonnet 4.5",
          "video": "assets/samples/task10021_014/models/claude-sonnet-4-5.mp4"
        },
        {
          "key": "gemini-3-pro",
          "label": "Gemini 3 Pro",
          "video": "assets/samples/task10021_014/models/gemini-3-pro.mp4"
        },
        {
          "key": "gpt-4.1",
          "label": "GPT-4.1",
          "video": "assets/samples/task10021_014/models/gpt-4.1.mp4"
        },
        {
          "key": "qwen3-vl-plus",
          "label": "Qwen3-VL-Plus",
          "video": "assets/samples/task10021_014/models/qwen3-vl-plus.mp4"
        }
      ]
    },
    {
      "id": "task10034_019",
      "label": "Case 8 (2D)",
      "prompt": "The scene consists of a mostly empty white background with no visible ground plane, platforms, or walls explicitly drawn; the environment appears to be an abstract, gravity-dominated space. Near the bottom-right region are four tall, thin, vertical light-gray bars arranged roughly in a row; two of these front bars are crossed by two diagonal light-gray bars forming an “X” structure, resembling a fragile support or gate. There are no other static obstacles or supports visible outside this vertical-bar assembly.\n\nThere are four small black circular objects, one medium light-gray circular object, and one larger red circular object. The red ball is located toward the upper-left of the image, isolated from the cluster of other objects and appears to be moving diagonally down and to the right toward the central area. The light-gray ball is located slightly below and to the right of the image center, just above three of the black balls; it appears to be moving slightly downward and to the right. The three black balls below the gray ball are arranged along a gentle downward-right diagonal line, suggesting they are sequential positions of a single ball or multiple balls following a similar path falling under gravity. A fourth black ball is positioned slightly farther down and right from this diagonal group, closer to the vertical-bar assembly, indicating continued motion in the same general direction.\n\nThe critical interactions are expected to involve gravity pulling all balls downward while their initial horizontal velocities carry them to the right, producing arcing trajectories. The red ball will likely fall and travel toward the cluster of darker objects, potentially colliding with the gray or black balls and altering their speeds and directions via elastic or partially inelastic collisions. The gray and black balls, already descending, may successively collide with each other along their diagonal line, transferring momentum and causing some to accelerate toward the lower-right vertical-bar structure. Ultimately, one or more balls may impact the crossed “X” support or adjacent vertical bars, possibly knocking parts of this structure aside or causing them to topple, illustrating a cause-effect chain of falling, collisions, and subsequent structural disturbance.",
      "gt_frames": {
        "first": "assets/samples/task10034_019/frame_01.png",
        "tenth": "assets/samples/task10034_019/frame_10.png"
      },
      "gt_video": "assets/samples/task10034_019/gt.mp4",
      "models": [
        {
          "key": "gpt-5",
          "label": "GPT-5",
          "video": "assets/samples/task10034_019/models/gpt-5.mp4"
        },
        {
          "key": "claude-sonnet-4-5",
          "label": "Claude Sonnet 4.5",
          "video": "assets/samples/task10034_019/models/claude-sonnet-4-5.mp4"
        },
        {
          "key": "gemini-3-pro",
          "label": "Gemini 3 Pro",
          "video": "assets/samples/task10034_019/models/gemini-3-pro.mp4"
        },
        {
          "key": "gpt-4.1",
          "label": "GPT-4.1",
          "video": "assets/samples/task10034_019/models/gpt-4.1.mp4"
        },
        {
          "key": "qwen3-vl-plus",
          "label": "Qwen3-VL-Plus",
          "video": "assets/samples/task10034_019/models/qwen3-vl-plus.mp4"
        }
      ]
    },
    {
      "id": "task10058_000",
      "label": "Case 9 (2D)",
      "prompt": "The scene is set against a plain white background with minimal static structures. The only clear static obstacle is a solid black diagonal line located on the left side, slanting upward from lower-left toward upper-right. There is also a faint, nearly horizontal light-gray bar slightly right of center; its translucency suggests it is a static guide or rail rather than a moving object. No ground plane is explicitly drawn, and there are no visible walls or fixed supports other than these lines.\n\nDynamic objects consist of multiple black circular dots and one larger red circular object. The red circle is positioned in the upper-right quadrant of the frame, initially at rest or moving slowly. A cluster of five smaller black dots is arranged in a curved, roughly vertical arc near the center-left of the image, forming an implied trajectory that curves downward and slightly rightward. One additional black dot is located farther down and to the right of this cluster, extending that curved path. These black dots represent successive positions of a single small object moving under gravity along that curved track, possibly influenced by the slanted black line or gray bar as deflecting surfaces.\n\nThe critical interaction sequence likely begins with the small black object moving along the curved path indicated by the dotted configuration, falling under gravity and perhaps being deflected by contact with the diagonal black line or sliding along the light-gray bar. The trajectory suggests the object descends from an upper-left or central region, curves around, and continues downward-right, potentially moving toward the region beneath the red circle. Future frames may show collisions where the moving black object either bounces off the slanted line or impacts the red circle, transferring momentum. Gravity primarily drives the downward motion, while contact forces with the slanted line or gray rail redirect the path, producing the observed arc of discrete positions.",
      "gt_frames": {
        "first": "assets/samples/task10058_000/frame_01.png",
        "tenth": "assets/samples/task10058_000/frame_10.png"
      },
      "gt_video": "assets/samples/task10058_000/gt.mp4",
      "models": [
        {
          "key": "gpt-5",
          "label": "GPT-5",
          "video": "assets/samples/task10058_000/models/gpt-5.mp4"
        },
        {
          "key": "claude-sonnet-4-5",
          "label": "Claude Sonnet 4.5",
          "video": "assets/samples/task10058_000/models/claude-sonnet-4-5.mp4"
        },
        {
          "key": "gemini-3-pro",
          "label": "Gemini 3 Pro",
          "video": "assets/samples/task10058_000/models/gemini-3-pro.mp4"
        },
        {
          "key": "gpt-4.1",
          "label": "GPT-4.1",
          "video": "assets/samples/task10058_000/models/gpt-4.1.mp4"
        },
        {
          "key": "qwen3-vl-plus",
          "label": "Qwen3-VL-Plus",
          "video": "assets/samples/task10058_000/models/qwen3-vl-plus.mp4"
        }
      ]
    }
  ]
};
