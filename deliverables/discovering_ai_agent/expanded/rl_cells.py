"""Self-contained Vietnamese notebook cells for reproducible tabular Q-learning."""
from textwrap import dedent


def _rl_cell(title, source, explanation, discussion):
    return dict(title=title, source=dedent(source).strip(),
                explanation=explanation, discussion=discussion)


RL_CELLS = [
    _rl_cell(
        "RL 01 — Ba bản đồ và cấu hình tái lập",
        '''
        rl_maps = {
            "A_open": ("S....", ".H.H.", ".....", ".H.H.", "....G"),
            "B_detour": ("S..#.", "##.#.", "...#.", ".###.", "....G"),
            "C_risky": ("S....", ".HHH.", "...H.", "H#...", "...HG"),
        }
        rl_actions = ((-1, 0), (0, 1), (1, 0), (0, -1))
        rl_action_names = ("up", "right", "down", "left")
        rl_config = dict(episodes=1500, max_steps=60, alpha=0.25, gamma=0.95,
                         epsilon_start=1.0, epsilon_end=0.05, seeds=list(range(10)),
                         eval_episodes=50, random_episodes=200, robust_episodes=100,
                         slip_probability=0.1, goal_reward=10.0, hole_reward=-10.0,
                         step_reward=-0.05, collision_reward=-0.15,
                         train_seed_rule="1000*map_index + seed",
                         eval_seed_rule="100000 + 10000*map_index + 100*seed",
                         robust_seed_rule="300000 + 10000*map_index + 100*seed",
                         random_seed_rule="200000 + map_index",
                         maps={name: list(grid) for name, grid in rl_maps.items()})
        assert all(len(g) == 5 and all(len(row) == 5 for row in g)
                   and sum(row.count("S") for row in g) == 1
                   and sum(row.count("G") for row in g) == 1 for g in rl_maps.values())
        print("3 bản đồ 5×5; 10 seed/bản đồ; 1500 episode/seed; tối đa 60 bước.")
        display(pd.DataFrame([{"map": n, "rows": " / ".join(g)} for n, g in rl_maps.items()]))
        ''',
        "S là điểm xuất phát, G là đích, H là hố kết thúc thất bại, # là tường, dấu chấm là ô an toàn. A có nhiều lối đi; B buộc đi vòng; C đặt hố gần đường đi. Cấu hình và bản đồ được lưu cùng kết quả để kiểm tra lại thí nghiệm.",
        "Ba bản đồ là ba bài toán hữu hạn được chọn trước, không phải ba mẫu ngẫu nhiên đại diện cho mọi môi trường. Mỗi seed thay đổi chuỗi khám phá và phá hòa khi hành động có cùng giá trị. Mười seed giúp quan sát độ nhạy của quá trình học trong từng bản đồ; chúng không thay thế kiểm tra trên bản đồ mới. Giữ nguyên điểm đầu và điểm đích còn làm nhiệm vụ đơn giản hơn bài toán định vị tổng quát. Mọi kết luận sau đây vì vậy gắn với đúng cấu hình được công bố.",
    ),
    _rl_cell(
        "RL 02 — Chuyển trạng thái, thưởng và kiểm tra môi trường",
        '''
        def rl_step(grid, state, action, rng, slip=0.0):
            if rng.random() < slip:
                action = rng.choice([a for a in range(4) if a != action])
            dr, dc = rl_actions[action]
            nxt = (state[0] + dr, state[1] + dc)
            if not (0 <= nxt[0] < 5 and 0 <= nxt[1] < 5) or grid[nxt[0]][nxt[1]] == "#":
                return state, -0.15, False
            tile = grid[nxt[0]][nxt[1]]
            return nxt, (10.0 if tile == "G" else -10.0 if tile == "H" else -0.05), tile in "GH"

        def rl_greedy(values, rng):
            best = max(values)
            return rng.choice([a for a, value in enumerate(values) if value == best])

        def rl_target(reward, terminated, next_values, gamma):
            return reward if terminated else reward + gamma * max(next_values)

        rl_probe_rng = random.Random(21)
        assert rl_step(rl_maps["A_open"], (0, 0), 0, rl_probe_rng) == ((0, 0), -0.15, False)
        assert rl_step(rl_maps["A_open"], (4, 3), 1, rl_probe_rng) == ((4, 4), 10.0, True)
        assert rl_step(rl_maps["A_open"], (1, 0), 1, rl_probe_rng) == ((1, 1), -10.0, True)
        assert rl_target(10, True, [999] * 4, 0.95) == 10
        print("PASS: biên/tường, đích, hố và bootstrap bằng 0 ở trạng thái kết thúc.")
        ''',
        "Trượt hành động xảy ra với xác suất slip: hành động dự định được thay bằng một trong ba hành động còn lại, xác suất bằng nhau. Trong huấn luyện, slip = 0. Mọi phép phá hòa chọn ngẫu nhiên giữa các hành động cùng tốt, tránh thiên lệch do thứ tự up/right/down/left.",
        "Phân biệt kết thúc tự nhiên và ngắt do giới hạn bước là điều kiện để cập nhật đúng. Khi chạm G hoặc H, mục tiêu TD chỉ còn phần thưởng vừa nhận, không cộng giá trị tương lai. Khi chạm tường, agent vẫn ở trạng thái cũ và tiếp tục tương tác. Phạt va chạm lớn hơn phạt bước thường nhằm khuyến khích di chuyển hữu ích. Đây là lựa chọn thiết kế phần thưởng; thay mức phạt có thể đổi tốc độ học và hành vi, nên các giá trị được ghi rõ thay vì xem như thuộc tính tất yếu của Q-learning.",
    ),
    _rl_cell(
        "RL 03 — Oracle BFS tìm đường an toàn ngắn nhất",
        '''
        def rl_bfs(grid):
            queue = deque([((0, 0), [(0, 0)])])
            seen = {(0, 0)}
            while queue:
                state, path = queue.popleft()
                if grid[state[0]][state[1]] == "G":
                    return path
                for dr, dc in rl_actions:
                    nxt = (state[0] + dr, state[1] + dc)
                    valid = 0 <= nxt[0] < 5 and 0 <= nxt[1] < 5
                    if valid and grid[nxt[0]][nxt[1]] not in "#H" and nxt not in seen:
                        seen.add(nxt)
                        queue.append((nxt, path + [nxt]))
            raise ValueError("Bản đồ không có đường an toàn đến đích")

        rl_oracle_paths = {name: rl_bfs(grid) for name, grid in rl_maps.items()}
        rl_oracle_rows = [{"map": name, "shortest_safe_steps": len(path) - 1,
                           "oracle_return": 10 - 0.05 * (len(path) - 2)}
                          for name, path in rl_oracle_paths.items()]
        rl_oracle_df = pd.DataFrame(rl_oracle_rows)
        assert all(path[0] == (0, 0) and path[-1] == (4, 4)
                   for path in rl_oracle_paths.values())
        display(rl_oracle_df)
        ''',
        "BFS được phép biết toàn bộ bản đồ và loại hố khỏi đồ thị. Vì mỗi bước có cùng độ dài, BFS cung cấp số bước an toàn ngắn nhất. Giá trị này là mốc kiểm tra khả năng giải và chất lượng đường đi, không phải kết quả học từ trải nghiệm.",
        "Oracle và agent có quyền truy cập thông tin khác nhau: BFS đọc bản đồ, còn bảng Q được cải thiện bằng tương tác. So sánh số bước với oracle cho biết chính sách đã học có đi vòng hay không, nhưng không chứng minh agent tự suy luận được cấu trúc bản đồ. Trong môi trường xác định này, đường an toàn ngắn nhất cũng tối ưu hóa tổng thưởng đã chọn. Khi có trượt, một đường dài hơn nhưng xa hố có thể tốt hơn; đường BFS xác định không còn mặc nhiên là chính sách tối ưu cho bài toán nhiễu.",
    ),
    _rl_cell(
        "RL 04 — Hàm huấn luyện Q-learning và lịch epsilon",
        '''
        def rl_train(grid, seed, method="Q-learning", episodes=None):
            rng = random.Random(seed)
            count = rl_config["episodes"] if episodes is None else episodes
            q = {(r, c): [0.0] * 4 for r in range(5) for c in range(5)
                 if grid[r][c] not in "#H"}
            history = []
            gamma = 0.0 if method == "GAMMA_ZERO" else rl_config["gamma"]
            for episode in range(count):
                epsilon = max(0.05, 1.0 - 0.95 * episode / max(1, 0.7 * (count - 1)))
                if method == "NO_EXPLORATION":
                    epsilon = 0.0
                state, total, terminated = (0, 0), 0.0, False
                for steps in range(1, rl_config["max_steps"] + 1):
                    action = rng.randrange(4) if rng.random() < epsilon else rl_greedy(q[state], rng)
                    nxt, reward, terminated = rl_step(grid, state, action, rng)
                    target = rl_target(reward, terminated, q.get(nxt, [0.0] * 4), gamma)
                    q[state][action] += rl_config["alpha"] * (target - q[state][action])
                    state, total = nxt, total + reward
                    if terminated:
                        break
                success = grid[state[0]][state[1]] == "G"
                history.append(dict(episode=episode + 1, total_return=total, steps=steps,
                                    success=int(success), terminated=terminated,
                                    truncated=not terminated, epsilon=epsilon))
            return q, history

        rl_demo_q, rl_demo_history = rl_train(rl_maps["A_open"], 0, episodes=20)
        display(pd.DataFrame(rl_demo_history).tail(5))
        print("Giới hạn 60 bước: truncation; vẫn bootstrap ở cập nhật cuối nếu chưa terminal.")
        ''',
        "Cập nhật dùng Q(s,a) ← Q(s,a) + α[r + γ max Q(s′,a′) − Q(s,a)]. Nhánh terminal bỏ hoàn toàn bootstrap. Giới hạn 60 bước là ngắt thu thập dữ liệu của nhiệm vụ tiếp diễn, không phải trạng thái hấp thụ; vì thế bước cuối vẫn bootstrap nếu chưa gặp đích hoặc hố.",
        "Epsilon giảm tuyến tính từ 1 xuống 0,05 trong 70% đầu quá trình, rồi giữ nguyên. Alpha cố định giúp minh họa dễ đọc nhưng không thỏa trực tiếp mọi giả thiết hội tụ tiệm cận trong định lý Q-learning cổ điển. Báo cáo vì vậy đo chất lượng sau ngân sách hữu hạn, không khẳng định hội tụ lý thuyết. NO_EXPLORATION chỉ bỏ nhánh chọn hành động epsilon; phá hòa vẫn ngẫu nhiên và khởi tạo Q bằng 0 vẫn có thể tạo khám phá gián tiếp khi phần thưởng bước là âm.",
    ),
    _rl_cell(
        "RL 05 — Bộ đánh giá tách biệt và khoảng Wilson",
        '''
        def rl_rollout(grid, q, rng, slip=0.0, max_steps=None):
            state, total, path, terminated = (0, 0), 0.0, [(0, 0)], False
            limit = rl_config["max_steps"] if max_steps is None else max_steps
            for steps in range(1, limit + 1):
                action = rng.randrange(4) if q is None else rl_greedy(q.get(state, [0.0] * 4), rng)
                state, reward, terminated = rl_step(grid, state, action, rng, slip)
                total += reward
                path.append(state)
                if terminated:
                    break
            return dict(success=grid[state[0]][state[1]] == "G", steps=steps,
                        total_return=total, path=path, terminated=terminated,
                        truncated=not terminated)

        def rl_wilson(wins, n):
            z, p = 1.959963984540054, wins / n
            center, scale = p + z*z/(2*n), 1 + z*z/n
            spread = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n))
            return (center - spread)/scale, (center + spread)/scale

        def rl_evaluate(grid, q, n, seed, slip=0.0):
            rng = random.Random(seed)
            trials = [rl_rollout(grid, q, rng, slip) for _ in range(n)]
            wins = sum(t["success"] for t in trials)
            lengths = [t["steps"] for t in trials if t["success"]]
            lo, hi = rl_wilson(wins, n)
            return dict(success_rate=wins/n, mean_steps_success=statistics.mean(lengths) if lengths else None,
                        mean_return=statistics.mean(t["total_return"] for t in trials), n=n, wins=wins,
                        ci_low=lo, ci_high=hi, truncated_rate=sum(t["truncated"] for t in trials)/n)

        print("Đánh giá: epsilon=0, Q đóng băng; seed riêng; Wilson 95% cho tỷ lệ thành công.")
        display(pd.DataFrame([rl_evaluate(rl_maps["A_open"], None, 20, 80001)]))
        ''',
        "Một rollout không cập nhật Q. Khi không truyền bảng Q, hành động được chọn đều ngẫu nhiên. Số bước trung bình chỉ tính trên lượt thành công; nếu không lượt nào thành công, kết quả là None. Tổng thưởng tính trên tất cả lượt, nên vẫn phản ánh thất bại và hết thời gian.",
        "Khoảng Wilson mô tả bất định nhị thức của các lượt đánh giá dưới một chính sách đã cố định và cơ chế ngẫu nhiên cụ thể. Nó không bao trùm mọi bất định do chọn bản đồ, thiết kế phần thưởng hay huấn luyện. Khi gộp nhiều seed, các chính sách khác nhau tạo ra phân bố hỗn hợp; khoảng gộp chỉ là tóm tắt mô tả. Vì vậy cần đọc thêm độ phân tán giữa các seed, nhất là trong thử nghiệm trượt. Tỷ lệ thành công cao không đủ để kết luận đường đi ngắn nếu bỏ qua cột số bước.",
    ),
    _rl_cell(
        "RL 06 — Huấn luyện 30 mô hình chuẩn và đo hiệu năng",
        '''
        rl_models, rl_histories, rl_trained_rows = {}, {}, []
        rl_started = time.perf_counter()
        for rl_map_index, (rl_name, rl_grid) in enumerate(rl_maps.items()):
            for rl_seed in rl_config["seeds"]:
                rl_key = (rl_name, "Q-learning", rl_seed)
                rl_q, rl_history = rl_train(rl_grid, 1000 * rl_map_index + rl_seed)
                rl_models[rl_key], rl_histories[rl_key] = rl_q, rl_history
                rl_stats = rl_evaluate(rl_grid, rl_q, rl_config["eval_episodes"],
                                       100000 + 10000 * rl_map_index + 100 * rl_seed)
                rl_trained_rows.append(dict(map=rl_name, method="Q-learning", seed=rl_seed, **rl_stats))
        rl_train_df = pd.DataFrame(rl_trained_rows)
        print(f"30 mô hình; thời gian huấn luyện + đánh giá: {time.perf_counter()-rl_started:.2f} s")
        display(rl_train_df.groupby("map").agg(success_mean=("success_rate", "mean"),
                     success_min=("success_rate", "min"), success_max=("success_rate", "max"),
                     mean_steps_success=("mean_steps_success", "mean"), mean_return=("mean_return", "mean")))
        ''',
        "Mỗi bản đồ có 10 lần huấn luyện độc lập và mỗi mô hình được kiểm tra bằng 50 lượt với seed đánh giá riêng. Các dòng theo seed được giữ lại để không che mất trường hợp một lần học thất bại. Thời gian đo gồm huấn luyện và đánh giá, phụ thuộc máy thực thi.",
        "Chính sách được kiểm tra trên đúng môi trường đã huấn luyện, với điểm xuất phát không đổi. Đây là đo mức học thuộc một nhiệm vụ điều khiển hữu hạn, không phải bằng chứng tổng quát hóa. Trong môi trường xác định, nhiều lượt đánh giá có thể trùng hoàn toàn nếu không còn hành động đồng hạng. Khi đó tăng số lượt không bổ sung nhiều thông tin bằng tăng số seed huấn luyện hoặc đổi điều kiện môi trường. Các cột min và max theo seed giúp nhận diện phần nào hạn chế này trước khi diễn giải giá trị trung bình.",
    ),
    _rl_cell(
        "RL 07 — Baseline ngẫu nhiên và bảng so sánh với oracle",
        '''
        rl_reference_rows = []
        for rl_map_index, (rl_name, rl_grid) in enumerate(rl_maps.items()):
            rl_stats = rl_evaluate(rl_grid, None, rl_config["random_episodes"], 200000 + rl_map_index)
            rl_reference_rows.append(dict(map=rl_name, method="Random", seed=None, **rl_stats))
            rl_path = rl_oracle_paths[rl_name]
            rl_length = len(rl_path) - 1
            rl_reference_rows.append(dict(map=rl_name, method="BFS oracle", seed=None,
                success_rate=1.0, mean_steps_success=rl_length, mean_return=10 - 0.05*(rl_length-1),
                n=1, wins=1, ci_low=None, ci_high=None, truncated_rate=0.0))

        def rl_aggregate(rows):
            grouped = defaultdict(list)
            for row in rows:
                grouped[(row["map"], row["method"])].append(row)
            result = []
            for (name, method), group in grouped.items():
                n, wins = sum(r["n"] for r in group), sum(r["wins"] for r in group)
                lengths = [(r["mean_steps_success"], r["wins"]) for r in group if r["wins"]]
                lo, hi = (None, None) if method == "BFS oracle" else rl_wilson(wins, n)
                result.append(dict(map=name, method=method, success_rate=wins/n,
                    mean_steps_success=sum(x*w for x, w in lengths)/wins if wins else None,
                    mean_return=sum(r["mean_return"]*r["n"] for r in group)/n, n=n, wins=wins,
                    ci_low=lo, ci_high=hi,
                    truncated_rate=sum(r["truncated_rate"]*r["n"] for r in group)/n))
            return result

        rl_initial_summary = pd.DataFrame(rl_aggregate(rl_trained_rows + rl_reference_rows))
        display(rl_initial_summary[["map", "method", "success_rate", "mean_steps_success", "mean_return", "n"]])
        ''',
        "Random có 200 rollout trên mỗi bản đồ. BFS là một đường đi tính toán xác định nên được ghi n = 1 và không gán khoảng tin cậy giả. Các trung bình gộp của mô hình học được tính theo số lượt, còn số bước thành công được cân theo số lần thành công.",
        "Một baseline ngẫu nhiên giúp kiểm tra nhiệm vụ có quá dễ hay không và lượng cải thiện thu được từ học. Tuy nhiên Random cũng bị giới hạn 60 bước, vì thế tỷ lệ thành công phụ thuộc ngân sách này. BFS có lợi thế biết toàn bản đồ và không cần chi phí học, nên cột này là cận tham chiếu đường đi an toàn. Các hàng có n khác nhau phải được đọc cùng ngữ cảnh; n = 500 lượt Q-learning không có nghĩa có 500 mô hình độc lập, mà chỉ có 10 mô hình được đánh giá lặp lại.",
    ),
    _rl_cell(
        "RL 08 — Ablation bỏ epsilon và bỏ chiết khấu tương lai",
        '''
        rl_ablation_rows = []
        rl_ablation_started = time.perf_counter()
        for rl_method in ("NO_EXPLORATION", "GAMMA_ZERO"):
            for rl_map_index, (rl_name, rl_grid) in enumerate(rl_maps.items()):
                for rl_seed in rl_config["seeds"]:
                    rl_key = (rl_name, rl_method, rl_seed)
                    rl_q, rl_history = rl_train(rl_grid, 1000 * rl_map_index + rl_seed, rl_method)
                    rl_models[rl_key], rl_histories[rl_key] = rl_q, rl_history
                    rl_stats = rl_evaluate(rl_grid, rl_q, rl_config["eval_episodes"],
                        100000 + 10000 * rl_map_index + 100 * rl_seed)
                    rl_ablation_rows.append(dict(map=rl_name, method=rl_method, seed=rl_seed, **rl_stats))
        rl_ablation_df = pd.DataFrame(rl_ablation_rows)
        rl_summary_rows = rl_aggregate(rl_trained_rows + rl_reference_rows + rl_ablation_rows)
        rl_summary_df = pd.DataFrame(rl_summary_rows)
        print(f"60 mô hình ablation; {time.perf_counter()-rl_ablation_started:.2f} s")
        display(rl_summary_df.pivot(index="map", columns="method", values="success_rate"))
        ''',
        "Mỗi ablation giữ cùng bản đồ, ngân sách và danh sách seed. NO_EXPLORATION đặt epsilon bằng 0 nhưng vẫn phá hòa ngẫu nhiên; GAMMA_ZERO giữ lịch epsilon và đặt gamma = 0. Cùng seed là một thiết kế ghép cặp theo định danh; sau khi chính sách khác nhau, chuỗi trạng thái và mức tiêu thụ số ngẫu nhiên cũng khác.",
        "Gamma bằng 0 chỉ tối ưu phần thưởng tức thời: ngoài vùng gần đích, nhiều hành động an toàn có cùng mức thưởng bước nên khó truyền thông tin về đích về điểm đầu. Ngược lại, bỏ epsilon chưa chắc gây thất bại trên môi trường nhỏ vì Q khởi tạo 0 lạc quan so với thưởng âm và phá hòa ngẫu nhiên vẫn đưa agent sang ô mới. Thí nghiệm không ép kết quả theo kỳ vọng. Nếu NO_EXPLORATION đạt tốt, kết luận đúng là cấu hình này chưa chứng minh epsilon thiết yếu, không phải khám phá vô ích trong mọi bài toán.",
    ),
    _rl_cell(
        "RL 09 — Độ bền khi 10% hành động bị trượt",
        '''
        rl_robustness_rows = []
        for rl_map_index, (rl_name, rl_grid) in enumerate(rl_maps.items()):
            for rl_seed in rl_config["seeds"]:
                rl_q = rl_models[(rl_name, "Q-learning", rl_seed)]
                rl_stats = rl_evaluate(rl_grid, rl_q, rl_config["robust_episodes"],
                    300000 + 10000 * rl_map_index + 100 * rl_seed, rl_config["slip_probability"])
                rl_robustness_rows.append(dict(map=rl_name, method="Q-learning + slip 0.1",
                    seed=rl_seed, slip_probability=0.1, **rl_stats))
        rl_robust_df = pd.DataFrame(rl_robustness_rows)
        rl_robust_summary = pd.DataFrame(rl_aggregate(rl_robustness_rows))
        display(rl_robust_summary[["map", "success_rate", "ci_low", "ci_high", "mean_steps_success", "mean_return"]])
        display(rl_robust_df.groupby("map")["success_rate"].agg(["min", "median", "max", "std"]))
        print("Không huấn luyện lại; nhiễu thay hành động bằng một trong ba hướng còn lại.")
        ''',
        "Mỗi chính sách chuẩn được giữ nguyên và chạy 100 lượt có slip = 0,1. Bảng đầu gộp 1.000 lượt trên mỗi bản đồ; bảng sau cho phân bố tỷ lệ thành công của 10 seed. So sánh với điều kiện xác định để quan sát ảnh hưởng của sai lệch cơ cấu chấp hành.",
        "Đây là kiểm tra chuyển điều kiện động lực học trên bản đồ cũ, không phải kiểm tra trên bản đồ chưa thấy. Một hành động trượt có thể rơi vào hố, chạm tường hoặc chỉ tăng số bước; bố trí vật cản quyết định mức suy giảm. Khoảng Wilson gộp không phản ánh đầy đủ biến thiên giữa mô hình học, nên báo cáo thêm min, trung vị, max và độ lệch chuẩn của tỷ lệ theo seed. Chính sách không được học trong nhiễu; kết quả cũng không cho biết huấn luyện có nhiễu sẽ cải thiện đến mức nào.",
    ),
    _rl_cell(
        "RL 10 — Đọc bảng Q và đường đi của một mô hình",
        '''
        rl_example_name, rl_example_seed = "C_risky", 0
        rl_example_q = rl_models[(rl_example_name, "Q-learning", rl_example_seed)]
        rl_q_rows = []
        for rl_state, rl_values in sorted(rl_example_q.items()):
            if rl_maps[rl_example_name][rl_state[0]][rl_state[1]] == "G":
                continue
            rl_best = max(rl_values)
            rl_q_rows.append(dict(row=rl_state[0], col=rl_state[1],
                **{name: value for name, value in zip(rl_action_names, rl_values)},
                best_actions="/".join(rl_action_names[a] for a, v in enumerate(rl_values) if v == rl_best)))
        rl_q_df = pd.DataFrame(rl_q_rows)
        display(rl_q_df.round(3))
        rl_example_trial = rl_rollout(rl_maps[rl_example_name], rl_example_q, random.Random(410001))
        print("Đường đi minh họa:", rl_example_trial["path"])
        print("Thành công:", rl_example_trial["success"], "Số bước:", rl_example_trial["steps"])
        ''',
        "Bảng trình bày đủ bốn giá trị hành động tại mỗi ô an toàn của C, seed 0. Cột best_actions liệt kê mọi hành động đồng hạng theo giá trị thực, không dựa trên số đã làm tròn. Đường đi minh họa dùng seed đánh giá riêng và không được dùng để chọn mô hình.",
        "Giá trị Q lớn gợi ý tổng thưởng chiết khấu tốt dưới trải nghiệm đã có, không phải xác suất hành động đúng hay xác suất sống sót. Các ô ít ghé thăm có thể chứa ước lượng kém dù đường đi từ điểm đầu đã tốt. Làm tròn số chỉ phục vụ đọc bảng và không tham gia chọn hành động. Một bảng của seed 0 giúp giải thích cơ chế, nhưng không đại diện thay cho phân bố 10 seed. Khi đánh giá có trượt, chính sách vẫn đọc bảng này nên rủi ro thay đổi mà các giá trị không tự điều chỉnh.",
    ),
    _rl_cell(
        "RL 11 — Đồ thị học qua episode trên cả ba bản đồ",
        '''
        rl_figure_paths = []
        rl_fig, rl_axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
        for rl_row, rl_name in enumerate(rl_maps):
            for rl_col, rl_metric in enumerate(("success", "total_return")):
                rl_runs = np.array([[point[rl_metric] for point in rl_histories[(rl_name, "Q-learning", seed)]]
                                    for seed in rl_config["seeds"]], dtype=float)
                rl_window = 50
                rl_smooth = np.array([np.convolve(run, np.ones(rl_window)/rl_window, mode="valid")
                                      for run in rl_runs])
                rl_x = np.arange(rl_window, rl_config["episodes"] + 1)
                rl_ax = rl_axes[rl_row, rl_col]
                rl_ax.plot(rl_x, rl_smooth.mean(axis=0), color="#2563eb")
                rl_ax.fill_between(rl_x, np.quantile(rl_smooth, .1, axis=0),
                                   np.quantile(rl_smooth, .9, axis=0), alpha=.2, color="#2563eb")
                rl_ax.set_title(f"{rl_name}: {rl_metric}")
                rl_ax.set_xlabel("Training episode (window=50)")
                rl_ax.grid(alpha=.25)
        rl_fig.suptitle("Q-learning training: mean and seed 10–90% quantiles")
        rl_fig.tight_layout()
        rl_figure_path = ARTIFACTS / "rl_learning_curves.png"
        rl_fig.savefig(rl_figure_path, dpi=150, bbox_inches="tight")
        rl_figure_paths.append(str(rl_figure_path))
        plt.show()
        print("Đã lưu", rl_figure_path)
        ''',
        "Hai cột là thành công và tổng thưởng trong huấn luyện; mỗi hàng là một bản đồ. Cửa sổ trượt 50 episode giảm nhiễu. Đường giữa là trung bình 10 seed, dải màu là phân vị 10–90% giữa seed sau khi làm trơn, không phải khoảng tin cậy.",
        "Đồ thị huấn luyện có thể duy trì tỷ lệ thất bại dù chính sách greedy kiểm tra đạt tốt, vì epsilon cuối vẫn bằng 0,05 và hành động ngẫu nhiên có thể rơi vào hố. Không nên so trực tiếp một điểm trên đường học với tỷ lệ đánh giá rồi kết luận số liệu mâu thuẫn. Cửa sổ 50 episode còn làm mờ các thay đổi nhanh ở giai đoạn đầu. Dải phân vị hẹp trên 10 seed chỉ mô tả mức dao động quan sát được trong thiết kế này, không chứng minh ổn định đối với bản đồ hoặc phân bố nhiệm vụ khác.",
    ),
    _rl_cell(
        "RL 12 — Trực quan bản đồ, oracle và đường đi đã học",
        '''
        rl_fig, rl_axes = plt.subplots(1, 3, figsize=(14, 4.5))
        for rl_ax, (rl_name, rl_grid) in zip(rl_axes, rl_maps.items()):
            rl_colors = {"#": 0.0, "H": .25, ".": .75, "S": .9, "G": 1.0}
            rl_ax.imshow([[rl_colors[tile] for tile in row] for row in rl_grid],
                         cmap="RdYlGn", vmin=0, vmax=1)
            for rl_r in range(5):
                for rl_c in range(5):
                    rl_tile = rl_grid[rl_r][rl_c]
                    if rl_tile != ".":
                        rl_ax.text(rl_c, rl_r, rl_tile, ha="center", va="center", weight="bold")
            rl_oracle = rl_oracle_paths[rl_name]
            rl_trial = rl_rollout(rl_grid, rl_models[(rl_name, "Q-learning", 0)], random.Random(420001))
            rl_ax.plot([s[1]-.06 for s in rl_oracle], [s[0]-.06 for s in rl_oracle],
                       "--", color="#111827", label=f"BFS: {len(rl_oracle)-1} steps", linewidth=2)
            rl_ax.plot([s[1]+.06 for s in rl_trial["path"]], [s[0]+.06 for s in rl_trial["path"]],
                       "o-", color="#2563eb", markersize=3, label=f"Q seed 0: {rl_trial['steps']} steps")
            rl_ax.set_title(rl_name)
            rl_ax.set_xticks(range(5)); rl_ax.set_yticks(range(5))
            rl_ax.legend(fontsize=8, loc="lower center", bbox_to_anchor=(.5, -.3))
        rl_fig.suptitle("Fixed maps: S start, G goal, H terminal hole, # wall")
        rl_fig.tight_layout()
        rl_figure_path = ARTIFACTS / "rl_maps_paths.png"
        rl_fig.savefig(rl_figure_path, dpi=150, bbox_inches="tight")
        rl_figure_paths.append(str(rl_figure_path))
        plt.show()
        print("Đã lưu", rl_figure_path)
        ''',
        "Đường nét đứt là BFS; đường xanh là rollout của Q-learning seed 0. Hai đường được lệch tọa độ nhẹ để nhìn thấy khi trùng nhau. Nhãn báo số bước thực tế. Các vị trí ô không thay đổi; đường vẽ chỉ là minh họa của một rollout xác định.",
        "Hình bản đồ giúp giải thích vì sao cùng tỷ lệ thành công có thể đi kèm độ bền khác nhau: tường chặn chuyển động còn hố kết thúc lượt ngay. Một đường bám sát hố vẫn tốt trong động lực học xác định nhưng dễ hỏng khi hành động trượt. Đường seed 0 được chọn cố định từ đầu, tránh chọn hình đẹp nhất sau khi xem kết quả. Nếu đường học trùng BFS, đó là bằng chứng cụ thể về số bước tối ưu trên bản đồ này; kết luận không mở rộng sang các trạng thái đầu khác hoặc môi trường chưa được huấn luyện.",
    ),
    _rl_cell(
        "RL 13 — So sánh phương pháp, ablation và phân bố độ bền",
        '''
        rl_fig, rl_axes = plt.subplots(1, 3, figsize=(17, 4.8))
        rl_methods = ["Random", "Q-learning", "BFS oracle", "NO_EXPLORATION", "GAMMA_ZERO"]
        rl_x = np.arange(len(rl_maps))
        for rl_i, rl_method in enumerate(rl_methods):
            rl_rates = [rl_summary_df.query("map == @name and method == @rl_method")["success_rate"].iloc[0]
                        for name in rl_maps]
            rl_axes[0].bar(rl_x + (rl_i-2)*.15, rl_rates, .15, label=rl_method)
        rl_axes[0].set_xticks(rl_x, list(rl_maps)); rl_axes[0].set_ylim(0, 1.08)
        rl_axes[0].set_title("Deterministic evaluation success")
        rl_axes[0].legend(fontsize=7, loc="upper center", bbox_to_anchor=(.5, -.15))
        for rl_i, rl_method in enumerate(["Q-learning", "NO_EXPLORATION", "GAMMA_ZERO"]):
            rl_first = [statistics.mean(p["success"] for p in rl_histories[(name, rl_method, seed)][:100])
                        for name in rl_maps for seed in rl_config["seeds"]]
            rl_axes[1].scatter(np.full(len(rl_first), rl_i) + np.linspace(-.16,.16,len(rl_first)),
                               rl_first, alpha=.6, s=18)
            rl_axes[1].plot(rl_i, statistics.mean(rl_first), "k_", markersize=22)
        rl_axes[1].set_xticks(range(3), ["Q", "No epsilon", "Gamma=0"])
        rl_axes[1].set_ylim(-.04, 1.04); rl_axes[1].set_title("First 100 training episodes (30 seeds/maps)")
        for rl_i, rl_name in enumerate(rl_maps):
            rl_values = rl_robust_df.query("map == @rl_name")["success_rate"].to_numpy()
            rl_axes[2].scatter(rl_i + np.linspace(-.12,.12,10), rl_values, alpha=.75)
            rl_mean = rl_robust_summary.query("map == @rl_name").iloc[0]
            rl_axes[2].errorbar(rl_i, rl_mean.success_rate,
                yerr=[[rl_mean.success_rate-rl_mean.ci_low], [rl_mean.ci_high-rl_mean.success_rate]],
                fmt="k_", capsize=5, markersize=20)
        rl_axes[2].set_xticks(rl_x, list(rl_maps)); rl_axes[2].set_ylim(0, 1.04)
        rl_axes[2].set_title("Slip 0.1: seed rates + pooled Wilson 95%")
        for rl_ax in rl_axes:
            rl_ax.set_ylabel("Success rate"); rl_ax.grid(axis="y", alpha=.25)
        rl_fig.tight_layout()
        rl_figure_path = ARTIFACTS / "rl_comparison_ablation_robustness.png"
        rl_fig.savefig(rl_figure_path, dpi=150, bbox_inches="tight")
        rl_figure_paths.append(str(rl_figure_path))
        plt.show()
        print("Đã lưu", rl_figure_path)
        ''',
        "Hình gồm tỷ lệ thành công khi đánh giá xác định, hiệu quả 100 episode huấn luyện đầu và phân bố tỷ lệ khi trượt. Chấm giữa chỉ báo từng seed/bản đồ; gạch đen là trung bình. Ở ô phải, chấm là từng seed và thanh lỗi là Wilson gộp có ý nghĩa mô tả đã nêu.",
        "Hiệu suất cuối và tốc độ học trả lời hai câu hỏi khác nhau. Một phương pháp đạt tỷ lệ cuối tương đương vẫn có thể cần nhiều tương tác ban đầu hơn hoặc ngược lại. Ô giữa gộp 30 cặp seed–bản đồ nên chỉ cho cái nhìn tổng quan, không là phép kiểm định độc lập 30 nhiệm vụ. Các cấu hình cùng seed cũng không tạo ra cùng quỹ đạo. Đồ thị không dùng dấu sao ý nghĩa thống kê hay tuyên bố nhân quả rộng; bảng theo seed và bản đồ được lưu để người đọc có thể phân tích lại với giả thiết phù hợp.",
    ),
    _rl_cell(
        "RL 14 — Lưu Q heatmap, bảng dữ liệu và hồ sơ thí nghiệm",
        '''
        rl_fig, rl_axes = plt.subplots(1, 4, figsize=(13, 3.3))
        for rl_action, rl_ax in enumerate(rl_axes):
            rl_heat = np.full((5, 5), np.nan)
            for rl_state, rl_values in rl_example_q.items():
                if rl_maps[rl_example_name][rl_state[0]][rl_state[1]] != "G":
                    rl_heat[rl_state] = rl_values[rl_action]
            rl_image = rl_ax.imshow(rl_heat, cmap="viridis", vmin=-10, vmax=10)
            rl_ax.set_title(rl_action_names[rl_action]); rl_ax.set_xticks(range(5)); rl_ax.set_yticks(range(5))
        rl_fig.colorbar(rl_image, ax=list(rl_axes), shrink=.75, label="Q value")
        rl_fig.suptitle("C_risky, seed 0: Q values; holes, walls and goal masked")
        rl_figure_path = ARTIFACTS / "rl_qtable_heatmap.png"
        rl_fig.savefig(rl_figure_path, dpi=150, bbox_inches="tight")
        rl_figure_paths.append(str(rl_figure_path))
        plt.show()
        rl_exports = {"rl_summary.csv": rl_summary_df, "rl_trained_seeds.csv": rl_train_df,
            "rl_ablation_seeds.csv": rl_ablation_df, "rl_robustness_seeds.csv": rl_robust_df,
            "rl_robustness_summary.csv": rl_robust_summary, "rl_oracle.csv": rl_oracle_df,
            "rl_qtable_example.csv": rl_q_df}
        for rl_filename, rl_table in rl_exports.items():
            rl_table.to_csv(ARTIFACTS / rl_filename, index=False, encoding="utf-8-sig")
        rl_history_rows = [dict(map=key[0], method=key[1], seed=key[2], **point)
                           for key, history in rl_histories.items() for point in history]
        pd.DataFrame(rl_history_rows).to_csv(ARTIFACTS / "rl_training_history.csv", index=False)
        rl_payload = dict(summary=rl_summary_rows, trained_rows=rl_trained_rows,
            ablation_rows=rl_ablation_rows, robustness_rows=rl_robustness_rows,
            robustness_summary=rl_robust_summary.to_dict(orient="records"), config=rl_config,
            oracle_rows=rl_oracle_rows, figure_paths=rl_figure_paths,
            notes=["Fixed-map evaluation; no claim of unseen-map generalization.",
                   "Wilson intervals are conditional/descriptive; ten fitted seeds per map.",
                   "Truncation bootstraps; termination does not; evaluation never updates Q."])
        (ARTIFACTS / "rl_extended_metrics.json").write_text(
            json.dumps(rl_payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        print("Đã lưu JSON hợp lệ (không NaN), 8 CSV, 4 PNG trong", ARTIFACTS)
        display(rl_summary_df[["map", "method", "success_rate", "mean_steps_success", "mean_return", "n"]])
        ''',
        "Heatmap dùng cùng thang −10 đến 10 để so bốn hành động. Ô hố, tường và đích được che. CSV chứa kết quả gộp, từng seed, oracle, bảng Q minh họa và toàn bộ lịch sử 135.000 episode. JSON dùng null cho số bước thành công không xác định và từ chối NaN bằng allow_nan=False.",
        "Lưu cấu hình và dữ liệu theo seed giúp kiểm chứng các số trong báo cáo, nhưng chưa đủ để tái lập tuyệt đối trên mọi nền tảng nếu phiên bản thư viện thay đổi. Các hình chỉ là cách trình bày, CSV và JSON mới giữ số chưa làm tròn. Bảng Q minh họa không thay thế toàn bộ mô hình; quy trình tái lập chính là chạy lại notebook theo thứ tự với seed đã ghi. Kết quả thực nghiệm nên được đọc cùng giới hạn: ba bản đồ nhỏ, điểm đầu cố định, thưởng do người thiết kế và nhiễu chỉ ở giai đoạn đánh giá.",
    ),
]
