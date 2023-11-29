def nerf_config_yaz(poses):
    with open("/home/yigit/Metaworld/rgb/transforms.json", "w") as file:
        file.write(
            """
                       {
        "camera_angle_y": 0.78,
        "frames": [
            
                       """
        )

        for i in range(120 * 120 * 3):
            file.write(
                """
    {17}
        
            \"file_path\": \"/home/yigit/Metaworld/rgb/rgb{0}\",
            \"transform_matrix\": [
                [
                    {1}, {2}, {3}, {4}
                ],
                [
                    {5}, {6}, {7}, {8}
                ],
                [
                    {9}, {10}, {11}, {12}
                ],
                [
                    {13}, {14}, {15}, {16}
                ]
            ]
    {18},\n
""".format(
                    i,
                    poses[i % 3, 0, 0],
                    poses[i % 3, 0, 1],
                    poses[i % 3, 0, 2],
                    poses[i % 3, 0, 3],
                    poses[i % 3, 1, 0],
                    poses[i % 3, 1, 1],
                    poses[i % 3, 1, 2],
                    poses[i % 3, 1, 3],
                    poses[i % 3, 2, 0],
                    poses[i % 3, 2, 1],
                    poses[i % 3, 2, 2],
                    poses[i % 3, 2, 3],
                    poses[i % 3, 3, 0],
                    poses[i % 3, 3, 1],
                    poses[i % 3, 3, 2],
                    poses[i % 3, 3, 3],
                    "{",
                    "}",
                )
            )

        file.write("]}")


'''            
    with open('/home/yigit/nerf/data/nerf_synthetic/lego/transforms_val.json', 'w') as the_file:
        for i in range(80,99):
            the_file.write("""
    {17}
        
            \"file_path\": \"/home/yigit/nerf/data/nerf_synthetic/lego/val/rgb{0}\",
        \"rotation\": 0.031415926535897934,
            \"transform_matrix\": [
                [
                    {1}, {2}, {3}, {4}
                ],
                [
                    {5}, {6}, {7}, {8}
                ],
                [
                    {9}, {10}, {11}, {12}
                ],
                [
                    {13}, {14}, {15}, {16}
                ]
            ]
    {18},\n
""".format(i,
            poses[i,0,0],poses[i,0,1],poses[i,0,2],poses[i,0,3],
            poses[i,1,0],poses[i,1,1],poses[i,1,2],poses[i,1,3],
            poses[i,2,0],poses[i,2,1],poses[i,2,2],poses[i,2,3],
            poses[i,3,0],poses[i,3,1],poses[i,3,2],poses[i,3,3],
            '{', '}')
)
            
    with open('/home/yigit/nerf/data/nerf_synthetic/lego/transforms_test.json', 'w') as the_file:
        i = 99
        the_file.write("""
{17}
    
        \"file_path\": \"/home/yigit/nerf/data/nerf_synthetic/lego/test/rgb{0}\",
        \"rotation\": 0.031415926535897934,
        \"transform_matrix\": [
            [
                {1}, {2}, {3}, {4}
            ],
            [
                {5}, {6}, {7}, {8}
            ],
            [
                {9}, {10}, {11}, {12}
            ],
            [
                {13}, {14}, {15}, {16}
            ]
        ]
{18},\n
""".format(i,
        poses[i,0,0],poses[i,0,1],poses[i,0,2],poses[i,0,3],
        poses[i,1,0],poses[i,1,1],poses[i,1,2],poses[i,1,3],
        poses[i,2,0],poses[i,2,1],poses[i,2,2],poses[i,2,3],
        poses[i,3,0],poses[i,3,1],poses[i,3,2],poses[i,3,3],
        '{', '}')
)
'''
