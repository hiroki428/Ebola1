import numpy as np
import networkx as nx
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# --- 1. 定数・パラメータ設定 (現状を維持) ---
N_nodes = 200      # 各層のノード数
N_layers = 5       # 5層 (0:H, 1:M, 2:F, 3:C, 4:T)
N_total = N_nodes * N_layers
simulation_days = 300
snapshot_day = 90  # 可視化したいタイミング（日）

params = {
    'betas': np.array([0.6, 0.4, 0.8, 0.3, 0.2]),# 感染率: beta_F > beta_H > beta_M > beta_C > beta_T
    'beta_Ds': np.array([0.1, 0.2, 0.5, 0.05, 0.0]),# 遺体からの感染率 beta_D
    'omega': 0.4,     # 層間カップリング強度 ω
    'sigma': 1/7,     # 潜伏期間
    'gamma': 1/12,    # 回復期間
    'mu': 0.3,        # 死亡率
    'delta': 1/2,      # 埋葬期間
    'p_inter': 0.0005   # 層間リンクの接続確率
}

# --- 2. ネットワークと伝播行列の生成 ---
def generate_multiplex_matrices(n, layers, p):
    np.random.seed(42) # 再現性
    networks = [nx.barabasi_albert_graph(n, 3) for _ in range(layers)]
    A_list = [nx.to_numpy_array(g) for g in networks]

    # 1. 層内の Supra-adjacency matrix
    A_supra = np.block([[A_list[i] if i == j else np.zeros((n, n))
                         for j in range(layers)] for i in range(layers)])

    # 2. 感染者(I)用の重み付き伝播行列 (beta_l * A^(l))
    A_weighted = np.block([[A_list[i] * p['betas'][i] if i == j else np.zeros((n, n))
                            for j in range(layers)] for i in range(layers)])

    # 3. 遺体(D)用の重み付き伝播行列 (beta_D,l * F^(l))
    F_weighted = np.block([[A_list[i] * p['beta_Ds'][i] if i == j else np.zeros((n, n))
                            for j in range(layers)] for i in range(layers)])

    # 4. 層間カップリング行列 C
    C = np.zeros((n * layers, n * layers))
    for l in range(layers):
        for m in range(layers):
            if l != m:
                mask = np.random.rand(n, n) < p['p_inter']
                C[l*n:(l+1)*n, m*n:(m+1)*n] = mask.astype(float)

    return A_supra, A_weighted, F_weighted, C

A_supra, A_weighted, F_weighted, C_matrix = generate_multiplex_matrices(N_nodes, N_layers, params)

# --- 3. 微分方程式系 ---
def model(y, t, A_w, F_w, C, A_base, p):
    y_reshaped = y.reshape((5, N_total))
    S, E, I, R, D = y_reshaped

    # 有効次数（層内接触 + 層間結合）で正規化
    deg = np.sum(A_base + C, axis=1)
    deg[deg == 0] = 1

    # 感染力 lambda_i (式8に準拠)
    lambda_i = (np.dot(A_w, I) + np.dot(F_w, D) + p['omega'] * np.dot(C, I + D)) / deg

    dS = -S * lambda_i
    dE = S * lambda_i - p['sigma'] * E
    dI = p['sigma'] * E - (p['gamma'] + p['mu']) * I
    dR = p['gamma'] * I + p['delta'] * D
    dD = p['mu'] * I - p['delta'] * D

    return np.concatenate([dS, dE, dI, dR, dD])

# --- 4. シミュレーション実行 ---
y0 = np.zeros((5, N_total))
y0[0, :] = 1.0
y0[0, :5] -= 0.01
y0[2, :5] = 0.01  # 初期感染者を配置

t = np.linspace(0, simulation_days, simulation_days)
sol = odeint(model, y0.flatten(), t, args=(A_weighted, F_weighted, C_matrix, A_supra, params))

# --- 5. ネットワーク可視化 (円形配置 & 層内外エッジの色分け・数表示) ---

# 層内(Intra)と層間(Inter)のグラフを個別に作成
G_intra = nx.from_numpy_array(A_supra)
G_inter = nx.from_numpy_array(C_matrix)

# それぞれのエッジ数を計算
intra_layer_edges = G_intra.number_of_edges()
inter_layer_edges = G_inter.number_of_edges()

# ノード描画用に全体のグラフも保持
G_total = nx.from_numpy_array(A_supra + C_matrix)

plt.figure(figsize=(7, 7))
pos_total = {}
R_radius = 1.7  # クラスターを配置する大きな円の半径

layer_names = ['Household', 'Healthcare', 'Funeral', 'Community', 'Mobility']

for l in range(N_layers):
    # 各層の内部レイアウトを個別に計算
    g_layer = nx.barabasi_albert_graph(N_nodes, 3)
    pos_layer = nx.spring_layout(g_layer, seed=42)

    # クラスターを円形に配置するための角度と座標(オフセット)を計算
    theta = 2.0 * np.pi * l / N_layers
    offset = np.array([R_radius * np.cos(theta), R_radius * np.sin(theta)])

    for i in range(N_nodes):
        pos_total[l * N_nodes + i] = pos_layer[i] + offset

    # 各クラスターのそばにレイヤー名のラベルを配置
    label_offset = np.array([(R_radius+1) * np.cos(theta), (R_radius+1) * np.sin(theta)])
    plt.text(label_offset[0], label_offset[1], layer_names[l],
             fontsize=14, fontweight='bold', ha='center', va='center',
             bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))

nx.draw_networkx_nodes(G_total, pos_total, node_size=20, node_color="#3cb1ff", alpha=1.0)# ノードの描画
nx.draw_networkx_edges(G_inter, pos_total, alpha=0.5, edge_color="#C0C0C0", width=1.5)# Inter-layer (層間) のリンク
nx.draw_networkx_edges(G_intra, pos_total, alpha=0.5, edge_color="#919191", width=1.5)# Intra-layer (層内) のリンク

# タイトルに両方のエッジ数を表示
plt.title(f"Multiplex Network Structure\nIntra-layer Edges: {intra_layer_edges} | Inter-layer Edges: {inter_layer_edges}",
          fontsize=16, fontweight='bold', pad=20)
plt.axis('off')
plt.tight_layout()
plt.show()

# --- 6. プロット (全体と感染動態の分割) ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
ax1.tick_params(axis='both', labelsize=16)
ax2.tick_params(axis='both', labelsize=16)

ax1.axvline(x=snapshot_day, color='red', linestyle='--', label=f'Snapshot (Day {snapshot_day})', alpha=0.7, linewidth=4)
ax1.plot(t, np.sum(sol[:, 0*N_total:1*N_total], axis=1), label='Susceptible', linewidth=4)
ax1.plot(t, np.sum(sol[:, 1*N_total:2*N_total], axis=1), label='Exposed', linewidth=4)
ax1.plot(t, np.sum(sol[:, 2*N_total:3*N_total], axis=1), label='Infected', linewidth=4)
ax1.plot(t, np.sum(sol[:, 3*N_total:4*N_total], axis=1), label='Recovered', linewidth=4)
ax1.plot(t, np.sum(sol[:, 4*N_total:5*N_total], axis=1), label='DeadBody', linewidth=4)
ax1.set_title("Total Population Dynamics", fontsize=20)
ax1.set_xlabel("Time (days)", fontsize=20)
ax1.set_ylabel("Number of Individuals", fontsize=20)
ax1.legend(fontsize=16)
ax1.grid(True)

ax2.axvline(x=snapshot_day, color='red', linestyle='--', label=f'Snapshot (Day {snapshot_day})', alpha=0.7, linewidth=4)
ax2.plot(t, np.sum(sol[:, 1*N_total:2*N_total], axis=1), label='Exposed', linewidth=4)
ax2.plot(t, np.sum(sol[:, 2*N_total:3*N_total], axis=1), label='Infected', linewidth=4)
ax2.plot(t, np.sum(sol[:, 4*N_total:5*N_total], axis=1), label='DeadBody', linewidth=4)
ax2.set_title("Epidemic Dynamics (E, I, D)", fontsize=20)
ax2.set_xlabel("Time (days)", fontsize=20)
ax2.legend(fontsize=20)
ax2.grid(True)
plt.tight_layout()
plt.show()


# --- 7. 画像2: ネットワークスナップショット (任意の日数での状態可視化) ---
t_idx = np.argmin(np.abs(t - snapshot_day))

# 指定時刻での全状態を取得 shape: (5, N_total)
state_at_t = sol[t_idx].reshape((5, N_total))

# 数値計算の微小な誤差によるマイナス値を0にクリップして防ぐ
state_at_t = np.clip(state_at_t, 0, None)

# 各ノードごとにS, E, I, R, Dの合計で割って「確率（割合）」に変換
probs = state_at_t / np.sum(state_at_t, axis=0)

# argmaxではなく、確率に基づいて各ノードの表示ステータスをサンプリングする
dominant_state_idx = np.zeros(N_total, dtype=int)
np.random.seed(42) # 描画の再現性のため
for i in range(N_total):
    dominant_state_idx[i] = np.random.choice([0, 1, 2, 3, 4], p=probs[:, i])

state_colors = ['#3498db', '#e67e22', '#e74c3c', '#2ecc71', '#95a5a6'] # S, E, I, R, Dの色配列
node_colors_total = [state_colors[idx] for idx in dominant_state_idx]

fig, axes = plt.subplots(1, 5, figsize=(20, 6))
layer_names = ['Household', 'Healthcare', 'Funeral', 'Community', 'Mobility']

for l in range(N_layers):
    # 各層の内部エッジのみを取り出してグラフ化
    A_layer = A_supra[l*N_nodes:(l+1)*N_nodes, l*N_nodes:(l+1)*N_nodes]
    G_layer = nx.from_numpy_array(A_layer)
    ax = axes[l]

    idx_start = l * N_nodes
    idx_end = (l + 1) * N_nodes
    colors_layer = node_colors_total[idx_start:idx_end]

    pos_layer = nx.spring_layout(G_layer, seed=42)

    nx.draw_networkx_edges(G_layer, pos_layer, ax=ax, edge_color="#C0C0C0", alpha=0.6, width=1.5)
    nx.draw_networkx_nodes(G_layer, pos_layer, ax=ax, node_color=colors_layer, node_size=50, alpha=1.0)

    ax.set_title(f"{layer_names[l]} Layer\nat Day {snapshot_day}", fontsize=20, fontweight='bold')
    ax.axis('off')

# レジェンドの設定（誤解を防ぐため D と R の意味を明確化）
legends = [mpatches.Patch(color=state_colors[i], label=['S', 'E', 'I', 'Removed (R/B)', 'Unburied (D)'][i]) for i in range(5)]
fig.legend(handles=legends, loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=5, fontsize=20, frameon=True)
plt.tight_layout(rect=[0, 0, 1, 0.82])
plt.show()

# --- 8. 画像3: 各レイヤーの感染関わる人数 (E + I + D) の推移を重ねて可視化 ---
plt.figure(figsize=(10, 6))
layer_names = ['Household', 'Healthcare', 'Funeral', 'Community', 'Mobility']
colors = ['#ff7f0e', '#1f77b4', '#9467bd', '#2ca02c', '#d62728'] # 各層を識別しやすいカラー

for l in range(N_layers):
    # 各層のノードインデックス範囲を定義
    start_idx = l * N_nodes
    end_idx = (l + 1) * N_nodes

    # sol（ODEの解）から各コンパートメントの該当層のデータを抽出して合算
    E_layer = np.sum(sol[:, 1*N_total + start_idx : 1*N_total + end_idx], axis=1)
    I_layer = np.sum(sol[:, 2*N_total + start_idx : 2*N_total + end_idx], axis=1)
    D_layer = np.sum(sol[:, 4*N_total + start_idx : 4*N_total + end_idx], axis=1)

    # 感染に関わる総人数 (E + I + D)
    active_cases_layer = E_layer + I_layer + D_layer

    plt.plot(t, active_cases_layer, label=f'{layer_names[l]} Layer', linewidth=3, color=colors[l])

plt.axvline(x=snapshot_day, color='red', linestyle='--', label=f'Snapshot (Day {snapshot_day})', alpha=0.7, linewidth=2)
plt.title("Temporal Evolution of Active Cases (E + I + D) per Layer", fontsize=14, fontweight='bold', pad=15)
plt.xlabel("Time (days)", fontsize=12)
plt.ylabel("Number of Individuals (E + I + D)", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(fontsize=11, loc='upper right')
plt.tight_layout()
plt.show()