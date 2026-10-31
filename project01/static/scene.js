/* Phòng khám AI - cảnh 3D low-poly (three r128). Xuất đối tượng toàn cục `Clinic`. */
(function (global) {
  'use strict';
  var T = THREE;

  var ZONES = {
    low: { color: '#2f7bf0', label: 'Thấp' },
    ok: { color: '#16a34a', label: 'An toàn' },
    high: { color: '#f59e0b', label: 'Cao' },
    danger: { color: '#e5342d', label: 'Nguy hiểm' }
  };
  var TYPE_COLORS = {
    condition: '#e5483d', symptom: '#f59e0b', risk_factor: '#8b5cf6', test: '#2589e0',
    treatment: '#16a34a', lifestyle: '#0d9f8f', complication: '#d6336c'
  };
  var TYPE_LABELS = {
    condition: 'Bệnh', symptom: 'Triệu chứng', risk_factor: 'Yếu tố nguy cơ', test: 'Xét nghiệm',
    treatment: 'Điều trị', lifestyle: 'Lối sống', complication: 'Biến chứng'
  };
  var REL_VI = {
    raises_risk: 'làm tăng nguy cơ', progresses_to: 'tiến triển thành', symptom_of: 'là triệu chứng của',
    diagnoses: 'giúp chẩn đoán', treats: 'điều trị', prevents: 'giúp phòng ngừa', supports: 'hỗ trợ',
    leads_to: 'dẫn đến', causes: 'gây ra', reduces_risk: 'giảm nguy cơ', indicates: 'gợi ý', measures: 'đo', is_a: 'là một loại', treated_by: 'được điều trị bằng', diagnosed_by: 'được chẩn đoán bằng', complication_of: 'là biến chứng của', monitors: 'theo dõi', helps: 'giúp ích cho', prevented_by: 'được ngăn ngừa nhờ', risk_factor_for: 'là yếu tố nguy cơ của'
  };
  function relVi(r) { return REL_VI[r] || String(r || '').replace(/_/g, ' '); }

  var DEFAULT_AGENTS = {
    doctor: { name: 'Bác sĩ trưởng', role: 'Điều phối, kết luận', color: '#2589e0', pos: [0, -3.2], kind: 'doctor' },
    predictor: { name: 'Bác sĩ dữ liệu', role: 'Deep Learning', color: '#7060d8', pos: [-5.3, -3.0], kind: 'predictor' },
    knowledge: { name: 'Thủ thư tri thức', role: 'KB + RAG', color: '#0d9f8f', pos: [5.3, -3.0], kind: 'knowledge' },
    coach: { name: 'Huấn luyện viên', role: 'Reinforcement learning', color: '#f0603a', pos: [-4.7, 1.7], kind: 'coach' },
    guardian: { name: 'Hộ lý an toàn', role: 'Luật cảnh báo', color: '#e8a317', pos: [4.8, 1.6], kind: 'guardian' },
    reporter: { name: 'Thư ký báo cáo', role: 'Sinh hình và bảng', color: '#e8508a', pos: [-2.9, -3.6], kind: 'reporter' }
  };
  var FACE_POINT = new T.Vector3(0, 0, 5);
  var PATIENT_SPOT = new T.Vector3(1.8, 0, 0.8);
  var PATIENT_DOOR = new T.Vector3(5.7, 0, 4.7);

  /* ---------- tiện ích ---------- */
  var stage, canvas, renderer, scene, camera, controls, clock;
  var clinic, kgGroup, bubblesEl;
  var chars = {}, patient = null, fxList = [], monitors = [], boardState = { tex: null, ctx: null, canvas: null };
  var mode = 'clinic', tween = null, speakerId = null, started = false, W = 800, H = 600;
  var alertState = { until: 0, color: '#f59e0b', level: 'warn' };
  var siren = null, sirenLight = null, hemi = null;
  var kgRadius = 7, kgData = null, kgLayout2d = null, kgNodeMeshes = [], kgHighlight = null, kgSelected = null, kgEdgeLabels = [], onPick = null;
  var raycaster = new T.Raycaster(), mouse = new T.Vector2(), pointerDown = null, hoverId = null;
  var glucoseState = { value: 112, zone: 'ok', hour: 6, series: [] };

  function stdMat(color, o) {
    return new T.MeshStandardMaterial(Object.assign({ color: color, roughness: 0.62, metalness: 0.04 }, o || {}));
  }
  function add(parent, geo, mat, x, y, z, noShadow) {
    var m = new T.Mesh(geo, mat);
    m.position.set(x || 0, y || 0, z || 0);
    m.castShadow = !noShadow; m.receiveShadow = true;
    parent.add(m); return m;
  }
  function B(w, h, d) { return new T.BoxGeometry(w, h, d); }
  function Sp(r, a, b, c, d, e, f) { return new T.SphereGeometry(r, a || 20, b || 14, c, d, e, f); }
  function Cy(rt, rb, h, s, hs, oe, ts, tl) { return new T.CylinderGeometry(rt, rb, h, s || 20, hs, oe, ts, tl); }
  function shade(hex, l) { var c = new T.Color(hex); var hsl = {}; c.getHSL(hsl); c.setHSL(hsl.h, hsl.s, Math.max(0, Math.min(1, hsl.l + l))); return '#' + c.getHexString(); }
  function clamp(v, a, b) { return Math.max(a, Math.min(b, v)); }
  function lerp(a, b, t) { return a + (b - a) * t; }
  function ease(t) { return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2; }
  function angDiff(a, b) { var d = (b - a) % (Math.PI * 2); if (d > Math.PI) d -= Math.PI * 2; if (d < -Math.PI) d += Math.PI * 2; return d; }
  function mulberry(seed) { var a = seed >>> 0; return function () { a |= 0; a = a + 0x6D2B79F5 | 0; var t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }

  function canvasTex(w, h, draw) {
    var c = document.createElement('canvas'); c.width = w; c.height = h;
    var ctx = c.getContext('2d'); if (draw) draw(ctx, w, h);
    var tex = new T.CanvasTexture(c); tex.anisotropy = 4; tex.minFilter = T.LinearFilter;
    return { canvas: c, ctx: ctx, tex: tex };
  }
  function rrect(ctx, x, y, w, h, r) {
    ctx.beginPath(); ctx.moveTo(x + r, y); ctx.arcTo(x + w, y, x + w, y + h, r); ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r); ctx.arcTo(x, y, x + w, y, r); ctx.closePath();
  }
  var FONT = '"Segoe UI","Helvetica Neue",Arial,sans-serif';

  /* ---------- biển tên ---------- */
  function makeSign(name, role, color) {
    var ct = canvasTex(512, 170);
    function draw(n, r) {
      var ctx = ct.ctx; ctx.clearRect(0, 0, 512, 170);
      ctx.fillStyle = '#ffffff'; rrect(ctx, 6, 6, 500, 158, 26); ctx.fill();
      ctx.lineWidth = 8; ctx.strokeStyle = color; rrect(ctx, 6, 6, 500, 158, 26); ctx.stroke();
      ctx.fillStyle = color; ctx.beginPath(); ctx.arc(52, 85, 22, 0, 7); ctx.fill();
      ctx.fillStyle = '#16213e'; ctx.font = 'bold 50px ' + FONT; ctx.textBaseline = 'middle';
      ctx.fillText(n, 90, 66, 400);
      ctx.fillStyle = shade(color, -0.22); ctx.font = 'bold 31px ' + FONT; ctx.fillText(r, 90, 122, 400);
      ct.tex.needsUpdate = true;
    }
    draw(name, role);
    var m = new T.Mesh(new T.PlaneGeometry(1.7, 0.565), new T.MeshBasicMaterial({ map: ct.tex, transparent: true }));
    m.userData.redraw = draw; return m;
  }

  /* ---------- robot ---------- */
  function buildRobot(id, hex, kind) {
    var root = new T.Group(), rig = new T.Group(); root.add(rig);
    var u = { rig: rig, mats: [], eyes: [], blinkAt: 2, blink: 0, phase: Math.random() * 6, color: hex };
    function glowMat(c, o) { var m = stdMat(c, Object.assign({ roughness: 0.38, emissive: hex, emissiveIntensity: 0.04 }, o || {})); u.mats.push(m); return m; }
    var main = glowMat(hex), white = glowMat('#ffffff'), light = glowMat(shade(hex, 0.25)), dark = stdMat('#25304a', { roughness: 0.5 });
    var skin = glowMat('#fff3e6');
    var bw = 0.62, headY = 2.12, shoulderY = 1.5;

    [-1, 1].forEach(function (s) { add(rig, Cy(0.13, 0.15, 0.42, 12), dark, s * 0.25, 0.3, 0); add(rig, B(0.36, 0.14, 0.5), main, s * 0.25, 0.07, 0.06); });

    var head = new T.Group(); head.position.set(0, headY, 0); rig.add(head); u.head = head;

    if (kind === 'doctor') {
      add(rig, Cy(0.52, 0.62, 1.1, 24), main, 0, 1.0, 0);
      add(rig, Cy(0.5, 0.6, 1.1, 24, 1, false, -0.9, 1.8), white, 0, 1.0, 0.02);
      add(rig, Cy(0.54, 0.54, 0.12, 24), main, 0, 1.52, 0);
      add(rig, B(0.5, 0.06, 0.04), main, 0, 1.18, 0.57).rotation.z = 0;
      var cross = add(rig, B(0.26, 0.08, 0.04), stdMat('#e5342d'), 0.28, 1.35, 0.55); add(rig, B(0.08, 0.26, 0.04), stdMat('#e5342d'), 0.28, 1.35, 0.55);
      var st = add(rig, new T.TorusGeometry(0.3, 0.035, 8, 24), dark, 0, 1.45, 0.3); st.rotation.x = 1.2;
      add(head, Sp(0.5, 24, 16), skin, 0, 0, 0);
      add(head, Cy(0.52, 0.52, 0.14, 24), main, 0, 0.3, 0);  // viền mũ
      add(head, Sp(0.5, 24, 10, 0, Math.PI * 2, 0, Math.PI / 2.4), white, 0, 0.12, 0).scale.set(1.04, 0.95, 1.04);
      var mir = add(head, Cy(0.14, 0.14, 0.03, 20), stdMat('#d8e6f2', { metalness: 0.7, roughness: 0.2 }), 0, 0.32, 0.48); mir.rotation.x = Math.PI / 2;
    } else if (kind === 'predictor') {
      add(rig, B(1.1, 1.05, 0.8), main, 0, 1.0, 0);
      add(rig, B(0.82, 0.62, 0.05), dark, 0, 1.05, 0.42);
      [0.3, 0.55, 0.8].forEach(function (h, i) { add(rig, B(0.16, h * 0.5, 0.05), glowMat(['#7fe3ff', '#ffe27f', '#ff9db5'][i], { emissiveIntensity: 0.5, emissive: ['#7fe3ff', '#ffe27f', '#ff9db5'][i] }), -0.25 + i * 0.25, 0.84 + h * 0.25, 0.46); });
      add(head, B(1.0, 0.8, 0.84), main, 0, 0, 0);
      add(head, B(0.8, 0.52, 0.06), dark, 0, 0.02, 0.43);
      add(head, Cy(0.03, 0.03, 0.4, 8), dark, 0, 0.6, 0);
      u.antenna = add(head, Sp(0.1, 14, 10), glowMat('#ffe27f', { emissive: '#ffcc33', emissiveIntensity: 0.6 }), 0, 0.84, 0);
      [-1, 1].forEach(function (s) { add(head, B(0.1, 0.3, 0.3), light, s * 0.55, 0, 0); });
    } else if (kind === 'knowledge') {
      var bd = add(rig, Sp(0.7, 24, 16), main, 0, 1.0, 0); bd.scale.set(1, 0.95, 0.85);
      var bl = add(rig, Sp(0.46, 20, 14), white, 0, 0.92, 0.3); bl.scale.set(1, 1, 0.55);
      add(head, Sp(0.52, 24, 16), light, 0, 0, 0).scale.set(1.08, 0.95, 0.95);
      [-1, 1].forEach(function (s) { var g = add(head, new T.TorusGeometry(0.17, 0.035, 8, 20), dark, s * 0.21, 0.07, 0.46); g.rotation.y = 0; });
      add(head, B(0.1, 0.03, 0.03), dark, 0, 0.07, 0.47);
      add(head, B(0.95, 0.07, 0.95), dark, 0, 0.5, 0).rotation.y = 0.4;
      add(head, Cy(0.28, 0.3, 0.16, 4), dark, 0, 0.42, 0).rotation.y = 0.4;
      var tas = add(head, Sp(0.07, 10, 8), glowMat('#ffd35c'), 0.52, 0.36, 0.2);
      var bk = new T.Group(); bk.position.set(0.0, 1.05, 0.62); rig.add(bk); u.book = bk;
      add(bk, B(0.62, 0.46, 0.1), stdMat('#ffcf5a'), 0, 0, 0); add(bk, B(0.56, 0.4, 0.04), stdMat('#fffaf0'), 0, 0, 0.07);
    } else if (kind === 'coach') {
      add(rig, Cy(0.42, 0.55, 1.15, 24), main, 0, 1.02, 0);
      add(rig, B(0.34, 0.9, 0.04), white, 0, 1.02, 0.52);
      add(rig, B(0.8, 0.12, 0.62), white, 0, 1.5, 0).scale.set(1, 1, 1);
      add(head, Sp(0.47, 24, 16), skin, 0, 0, 0);
      var hb = add(head, new T.TorusGeometry(0.46, 0.07, 10, 28), main, 0, 0.14, 0); hb.rotation.x = Math.PI / 2;
      add(head, Sp(0.1, 10, 8), white, 0.42, 0.14, 0.12);
      var wh = add(rig, Cy(0.07, 0.07, 0.12, 10), glowMat('#ffe27f'), -0.2, 1.4, 0.45); wh.rotation.z = Math.PI / 2;
      var dumb = new T.Group(); u.dumbbell = dumb; dumb.position.set(0, -0.62, 0.1);
      add(dumb, Cy(0.04, 0.04, 0.5, 8), dark, 0, 0, 0).rotation.z = Math.PI / 2;
      [-1, 1].forEach(function (s) { add(dumb, Cy(0.14, 0.14, 0.1, 14), dark, s * 0.24, 0, 0).rotation.z = Math.PI / 2; });
    } else if (kind === 'guardian') {
      add(rig, B(1.15, 1.1, 0.75), main, 0, 1.0, 0);
      var sh = add(rig, Cy(0.36, 0.36, 0.06, 6), white, 0, 1.05, 0.4); sh.rotation.x = Math.PI / 2; sh.rotation.y = Math.PI / 6;
      add(rig, B(0.12, 0.34, 0.04), main, 0, 1.05, 0.44); add(rig, B(0.34, 0.12, 0.04), main, 0, 1.05, 0.44);
      add(head, Sp(0.48, 24, 16), skin, 0, 0, 0);
      var hat = add(head, Sp(0.52, 24, 12, 0, Math.PI * 2, 0, Math.PI / 2), main, 0, 0.12, 0);
      add(head, Cy(0.58, 0.58, 0.06, 24), main, 0, 0.14, 0.04);
      add(head, B(0.12, 0.5, 0.04), white, 0, 0.3, 0.02).rotation.x = -0.4;
    } else { // reporter
      add(rig, Cy(0.38, 0.62, 1.1, 24), main, 0, 1.0, 0);
      add(rig, Cy(0.4, 0.4, 0.1, 24), white, 0, 1.5, 0);
      add(head, Sp(0.48, 24, 16), skin, 0, 0, 0);
      add(head, Cy(0.5, 0.52, 0.12, 24), main, 0.06, 0.4, 0).rotation.z = -0.12;
      add(head, Sp(0.42, 20, 8, 0, Math.PI * 2, 0, Math.PI / 2), main, 0.06, 0.38, 0).scale.set(1.15, 0.7, 1.15);
      var pen = add(head, Cy(0.025, 0.025, 0.4, 8), glowMat('#ffd35c'), -0.46, 0.05, 0.1); pen.rotation.z = 0.5;
      var cb = new T.Group(); cb.position.set(0.0, 1.0, 0.55); rig.add(cb); u.book = cb;
      add(cb, B(0.5, 0.64, 0.05), white, 0, 0, 0); add(cb, B(0.4, 0.5, 0.02), stdMat('#ffe6ef'), 0, 0, 0.04);
      [0.15, 0, -0.15].forEach(function (y) { add(cb, B(0.3, 0.03, 0.02), main, 0, y, 0.06); });
    }

    /* mắt + miệng */
    var eyeMat = kind === 'predictor' ? stdMat('#7fe3ff', { emissive: '#4fd8ff', emissiveIntensity: 0.9 }) : stdMat('#1d2740', { roughness: 0.3 });
    var ez = kind === 'predictor' ? 0.47 : kind === 'knowledge' ? 0.47 : 0.43;
    [-1, 1].forEach(function (s) {
      var e = new T.Mesh(Sp(0.075, 12, 10), eyeMat); e.position.set(s * 0.19, 0.06, ez); e.scale.set(1, 1.35, 0.6); head.add(e); u.eyes.push(e);
      if (kind !== 'predictor') {
        var h = new T.Mesh(Sp(0.025, 8, 6), new T.MeshBasicMaterial({ color: '#ffffff' })); h.position.set(s * 0.19 + 0.02, 0.1, ez + 0.04); head.add(h);
        var ch = new T.Mesh(Sp(0.06, 10, 8), stdMat('#ff9db0', { roughness: 0.9 })); ch.position.set(s * 0.3, -0.07, ez - 0.07); ch.scale.set(1.2, 0.7, 0.4); head.add(ch);
      }
    });
    var mouth = new T.Mesh(Sp(0.06, 10, 8), stdMat(kind === 'predictor' ? '#7fe3ff' : '#8b2d4a', kind === 'predictor' ? { emissive: '#4fd8ff', emissiveIntensity: 0.7 } : {}));
    mouth.position.set(0, -0.12, ez - 0.01); mouth.scale.set(1.5, 0.5, 0.5); head.add(mouth); u.mouth = mouth;

    /* tay */
    u.arms = [];
    [-1, 1].forEach(function (s) {
      var piv = new T.Group(); piv.position.set(s * (kind === 'predictor' || kind === 'guardian' ? 0.68 : kind === 'knowledge' ? 0.72 : 0.6), shoulderY, 0); rig.add(piv);
      add(piv, Cy(0.1, 0.09, 0.6, 12), kind === 'doctor' ? white : main, 0, -0.3, 0);
      add(piv, Sp(0.13, 12, 10), kind === 'doctor' ? skin : light, 0, -0.64, 0);
      u.arms.push(piv);
    });
    if (u.dumbbell) u.arms[1].add(u.dumbbell);

    /* vòng sáng dưới chân + vệ tinh "đang làm việc" */
    var ring = new T.Mesh(new T.RingGeometry(0.85, 1.05, 40), new T.MeshBasicMaterial({ color: hex, transparent: true, opacity: 0, side: T.DoubleSide, depthWrite: false }));
    ring.rotation.x = -Math.PI / 2; ring.position.y = 0.03; root.add(ring); u.ring = ring;
    var orb = new T.Group(); orb.position.set(0, headY + 0.85, 0); orb.visible = false; rig.add(orb); u.orb = orb;
    for (var i = 0; i < 3; i++) { var o = new T.Mesh(Sp(0.08, 10, 8), new T.MeshBasicMaterial({ color: hex })); o.position.set(Math.cos(i * 2.09) * 0.38, 0, Math.sin(i * 2.09) * 0.38); orb.add(o); }

    root.userData = u; return root;
  }

  /* ---------- bệnh nhân ---------- */
  function buildPatient() {
    var root = new T.Group(), rig = new T.Group(); root.add(rig);
    var u = { rig: rig, eyes: [], blinkAt: 3, blink: 0, phase: 1, color: '#2589e0' };
    var skinM = stdMat('#ffd9b8'), hairM = stdMat('#3b2a22'), pantsM = stdMat('#3a4668'), shirtM = stdMat('#2589e0', { roughness: 0.55 });
    u.shirtM = shirtM; u.hairM = hairM;
    u.legs = [];
    [-1, 1].forEach(function (s) {
      var lp = new T.Group(); lp.position.set(s * 0.2, 0.85, 0); rig.add(lp);
      add(lp, Cy(0.15, 0.13, 0.8, 12), pantsM, 0, -0.4, 0); add(lp, B(0.3, 0.12, 0.48), stdMat('#fafafa'), 0, -0.82, 0.08);
      u.legs.push(lp);
    });
    var torso = new T.Group(); rig.add(torso); u.torso = torso;
    add(torso, Cy(0.44, 0.5, 1.0, 20), shirtM, 0, 1.3, 0);
    var belly = add(torso, Sp(0.5, 18, 12), shirtM, 0, 1.0, 0.1); belly.scale.set(0.9, 0.7, 0.6); u.belly = belly;
    var skirt = add(rig, Cy(0.5, 0.8, 0.42, 20), pantsM, 0, 0.94, 0); skirt.visible = false; u.skirt = skirt;
    u.arms = [];
    [-1, 1].forEach(function (s) {
      var piv = new T.Group(); piv.position.set(s * 0.58, 1.7, 0); torso.add(piv);
      add(piv, Cy(0.1, 0.09, 0.7, 12), shirtM, 0, -0.35, 0); add(piv, Sp(0.12, 12, 10), skinM, 0, -0.74, 0);
      u.arms.push(piv);
    });
    var food = new T.Group(); food.visible = false; food.position.set(0, -0.82, 0.08); u.arms[1].add(food); u.food = food;
    var head = new T.Group(); head.position.set(0, 2.38, 0); rig.add(head); u.head = head;
    add(head, Sp(0.46, 24, 16), skinM, 0, 0, 0);
    var hair = add(head, Sp(0.5, 24, 12, 0, Math.PI * 2, 0, Math.PI / 2.6), hairM, 0, 0.05, -0.07); u.hair = hair;
    var longHair = add(head, B(0.95, 1.0, 0.3), hairM, 0, -0.25, -0.34); longHair.visible = false; u.longHair = longHair;
    var bun = add(head, Sp(0.2, 12, 10), hairM, 0, 0.52, -0.1); bun.visible = false; u.bun = bun;
    [-1, 1].forEach(function (s) {
      var e = new T.Mesh(Sp(0.07, 12, 10), stdMat('#1d2740', { roughness: 0.3 })); e.position.set(s * 0.17, 0.03, 0.42); e.scale.set(1, 1.35, 0.6); head.add(e); u.eyes.push(e);
      var h = new T.Mesh(Sp(0.022, 8, 6), new T.MeshBasicMaterial({ color: '#fff' })); h.position.set(s * 0.17 + 0.02, 0.07, 0.46); head.add(h);
      var ch = new T.Mesh(Sp(0.06, 10, 8), stdMat('#ff9db0', { roughness: 0.9 })); ch.position.set(s * 0.28, -0.08, 0.36); ch.scale.set(1.2, 0.7, 0.4); head.add(ch);
    });
    var mouth = new T.Mesh(Sp(0.06, 10, 8), stdMat('#9b3350')); mouth.position.set(0, -0.14, 0.42); mouth.scale.set(1.4, 0.45, 0.5); head.add(mouth); u.mouth = mouth;
    var zz = canvasTex(256, 128, function (ctx) { ctx.fillStyle = '#2f5bd8'; ctx.font = 'bold 90px ' + FONT; ctx.textBaseline = 'middle'; ctx.fillText('Z z z', 20, 64); });
    var zs = new T.Sprite(new T.SpriteMaterial({ map: zz.tex, transparent: true })); zs.scale.set(1.2, 0.6, 1); zs.position.set(0.7, 3.1, 0); zs.visible = false; rig.add(zs); u.zzz = zs;
    root.userData = u;
    root.userData.state = 'idle'; root.userData.stateT = 0; root.userData.move = null; root.userData.pose = { aLx: 0, aRx: 0, aLz: -0.12, aRz: 0.12, lLx: 0, lRx: 0, lLz: 0, lRz: 0, y: 0, rz: 0, hx: 0, hz: 0 };
    root.userData.foodKind = 'normal_meal';
    return root;
  }

  function setFood(u, kind) {
    var f = u.food; while (f.children.length) f.remove(f.children[0]);
    if (kind === 'sweet') { add(f, Cy(0.16, 0.16, 0.09, 16), stdMat('#f4a6c8'), 0, 0.05, 0.06); add(f, Cy(0.17, 0.17, 0.05, 16), stdMat('#ff5fa2'), 0, 0.1, 0.06); add(f, Sp(0.04, 8, 6), stdMat('#fff'), 0.06, 0.14, 0.06); }
    else if (kind === 'light_meal') { add(f, Sp(0.13, 12, 10), stdMat('#74c93c'), 0, 0.05, 0.05); add(f, Cy(0.01, 0.01, 0.1, 6), stdMat('#5a3a1a'), 0, 0.2, 0.05); }
    else if (kind === 'heavy_meal') { add(f, Cy(0.2, 0.2, 0.08, 16), stdMat('#d9a05b'), 0, 0.05, 0.08); add(f, Cy(0.21, 0.21, 0.07, 16), stdMat('#6b3a1e'), 0, 0.12, 0.08); add(f, Cy(0.2, 0.2, 0.03, 16), stdMat('#7bd13b'), 0, 0.16, 0.08); add(f, Cy(0.2, 0.2, 0.08, 16), stdMat('#d9a05b'), 0, 0.22, 0.08); }
    else { add(f, Cy(0.2, 0.16, 0.08, 16), stdMat('#ffffff'), 0, 0.02, 0.08); add(f, Sp(0.16, 14, 8, 0, Math.PI * 2, 0, Math.PI / 2), stdMat('#fff4dd'), 0, 0.05, 0.08); }
  }

  var ACTION_STATE = { rest: 'rest', sweet: 'eat', light_meal: 'eat', normal_meal: 'eat', heavy_meal: 'eat', walk: 'walk', workout: 'workout', skip: 'skip' };

  function updatePatient(t, dt) {
    var u = patient.userData, p = u.pose, s = u.state;
    u.stateT += dt;
    var tp = { aLx: 0, aRx: 0, aLz: -0.12, aRz: 0.12, lLx: 0, lRx: 0, lLz: 0, lRz: 0, y: Math.sin(t * 1.8 + 1) * 0.015, rz: 0, hx: 0, hz: 0 };
    var closed = false;
    if (s === 'walk') {
      var w = Math.sin(u.stateT * 7);
      tp.lLx = w * 0.7; tp.lRx = -w * 0.7; tp.aLx = -w * 0.6; tp.aRx = w * 0.6; tp.y = Math.abs(Math.sin(u.stateT * 7)) * 0.07;
    } else if (s === 'eat') {
      var e = 0.5 + 0.5 * Math.sin(u.stateT * 5.5);
      tp.aRx = -1.35 - 0.8 * e; tp.aRz = 0.1; tp.hx = 0.1 * e; tp.aLx = -0.4;
    } else if (s === 'workout') {
      var ph = Math.sin(u.stateT * 6.5), ab = 0.5 + 0.5 * ph;
      tp.aLz = -(0.2 + 2.6 * ab); tp.aRz = 0.2 + 2.6 * ab; tp.lLz = -0.45 * ab; tp.lRz = 0.45 * ab; tp.y = Math.abs(ph) * 0.38;
    } else if (s === 'rest') {
      var st = 0.5 + 0.5 * Math.sin(u.stateT * 1.6);
      tp.aLz = -(0.3 + 2.3 * st); tp.aRz = 0.3 + 2.3 * st; tp.hx = 0.15; tp.hz = 0.12 * Math.sin(u.stateT * 1.2); tp.y = -0.04; closed = true;
    } else if (s === 'skip') {
      tp.aLz = -0.7; tp.aRz = 0.7; tp.y = Math.max(0, Math.sin(u.stateT * 5)) * 0.05; tp.hz = 0.18 * Math.sin(u.stateT * 3.5); tp.rz = 0.05 * Math.sin(u.stateT * 3.5);
    } else if (s === 'wave') {
      tp.aRz = 2.4 + 0.4 * Math.sin(u.stateT * 8);
    }
    var k = Math.min(1, dt * 9);
    Object.keys(tp).forEach(function (key) { p[key] = lerp(p[key], tp[key], k); });
    u.arms[0].rotation.x = p.aLx; u.arms[1].rotation.x = p.aRx;
    u.arms[0].rotation.z = p.aLz; u.arms[1].rotation.z = p.aRz;
    u.legs[0].rotation.x = p.lLx; u.legs[1].rotation.x = p.lRx; u.legs[0].rotation.z = p.lLz; u.legs[1].rotation.z = p.lRz;
    u.rig.position.y = p.y; u.rig.rotation.z = p.rz;
    u.food.visible = (s === 'eat');
    u.zzz.visible = (s === 'rest'); if (u.zzz.visible) u.zzz.position.y = 3.1 + Math.sin(u.stateT * 2) * 0.12;
    u.closed = closed;
    // di chuyển
    if (u.move) {
      var m = u.move, dx = m.to.x - patient.position.x, dz = m.to.z - patient.position.z, d = Math.sqrt(dx * dx + dz * dz);
      if (d < 0.08) { var r = m.resolve; u.move = null; setPatientState('idle'); patient.rotation.y = m.faceYaw; if (r) r(); }
      else {
        var step = Math.min(d, m.speed * dt); patient.position.x += dx / d * step; patient.position.z += dz / d * step;
        var ty = Math.atan2(dx, dz); patient.rotation.y += angDiff(patient.rotation.y, ty) * Math.min(1, dt * 8);
      }
    }
  }
  function setPatientState(s) { var u = patient.userData; if (u.state !== s) { u.state = s; u.stateT = 0; } }

  /* ---------- cập nhật robot mỗi khung ---------- */
  function updateChar(c, t, dt) {
    var u = c.userData, speaking = c.userData.speaking, working = c.userData.working;
    var ph = t + u.phase;
    var bounce = Math.sin(ph * 1.7) * 0.025;
    var arm0 = -0.1 - Math.sin(ph * 1.3) * 0.03, arm1 = 0.1 + Math.sin(ph * 1.3 + 1) * 0.03;
    var ax0 = 0, ax1 = 0;
    var glow = 0.05;
    if (speaking) {
      bounce = Math.abs(Math.sin(t * 8)) * 0.12; glow = 0.38 + 0.2 * Math.sin(t * 6);
      arm1 = 0.9 + 0.5 * Math.sin(t * 5); ax1 = -0.5; arm0 = -0.25;
      u.mouth.scale.y = 0.4 + Math.abs(Math.sin(t * 13)) * 1.0;
    } else u.mouth.scale.y = lerp(u.mouth.scale.y, 0.5, 0.3);
    if (working) { glow = 0.25 + 0.15 * Math.sin(t * 9); }
    u.rig.position.y = lerp(u.rig.position.y, bounce, 0.4);
    u.arms[0].rotation.z = lerp(u.arms[0].rotation.z, arm0, 0.2); u.arms[1].rotation.z = lerp(u.arms[1].rotation.z, arm1, 0.2);
    u.arms[0].rotation.x = lerp(u.arms[0].rotation.x, ax0, 0.2); u.arms[1].rotation.x = lerp(u.arms[1].rotation.x, ax1, 0.2);
    u.mats.forEach(function (m) { m.emissiveIntensity = lerp(m.emissiveIntensity, glow, 0.25); });
    u.ring.material.opacity = lerp(u.ring.material.opacity, speaking ? 0.55 + 0.25 * Math.sin(t * 6) : 0, 0.2);
    u.ring.scale.setScalar(1 + (speaking ? 0.08 * Math.sin(t * 6) : 0));
    u.orb.visible = !!working; if (working) { u.orb.rotation.y += dt * 4; }
    if (u.antenna) u.antenna.material.emissiveIntensity = 0.5 + 0.4 * Math.sin(t * 5 + u.phase);
    if (u.book && speaking) u.book.rotation.z = Math.sin(t * 4) * 0.12; else if (u.book) u.book.rotation.z *= 0.9;
    // quay đầu
    var target;
    if (c.userData.look && c.userData.look !== c) {
      var dx = c.userData.look.position.x - c.position.x, dz = c.userData.look.position.z - c.position.z;
      target = angDiff(c.rotation.y, Math.atan2(dx, dz));
    } else target = Math.sin(t * 0.45 + u.phase * 2) * 0.35 * (Math.sin(t * 0.17 + u.phase) > 0 ? 1 : 0.3);
    u.head.rotation.y = lerp(u.head.rotation.y, clamp(target, -0.9, 0.9), 0.07);
    u.head.rotation.x = lerp(u.head.rotation.x, working ? -0.2 : (speaking ? 0.06 * Math.sin(t * 8) : 0), 0.12);
    // chớp mắt
    blinkEyes(u, t, false);
  }
  function blinkEyes(u, t, forceClosed) {
    if (t > u.blinkAt) { u.blink = t + 0.13; u.blinkAt = t + 2.3 + Math.random() * 3; }
    var shut = forceClosed || t < u.blink;
    u.eyes.forEach(function (e) { e.scale.y = lerp(e.scale.y, shut ? 0.12 : 1.35, 0.5); });
  }

  /* ---------- đồ vật phòng khám ---------- */
  function buildRoom() {
    var g = new T.Group(); clinic = g; scene.add(g);
    // sàn
    var floorTex = canvasTex(512, 512, function (ctx) {
      for (var i = 0; i < 8; i++) for (var j = 0; j < 8; j++) { ctx.fillStyle = (i + j) % 2 ? '#bfe8d6' : '#e3f6ee'; ctx.fillRect(i * 64, j * 64, 64, 64); }
      ctx.strokeStyle = '#a5d4c0'; ctx.lineWidth = 3; for (var k = 0; k <= 8; k++) { ctx.beginPath(); ctx.moveTo(k * 64, 0); ctx.lineTo(k * 64, 512); ctx.moveTo(0, k * 64); ctx.lineTo(512, k * 64); ctx.stroke(); }
    });
    floorTex.tex.wrapS = floorTex.tex.wrapT = T.RepeatWrapping; floorTex.tex.repeat.set(7, 6.5);
    var floor = new T.Mesh(new T.PlaneGeometry(14.4, 13.4), new T.MeshLambertMaterial({ map: floorTex.tex }));
    floor.rotation.x = -Math.PI / 2; floor.position.set(0, 0, 0.7); floor.receiveShadow = true; g.add(floor);
    // thảm tròn
    var rug = add(g, Cy(3.5, 3.5, 0.03, 48), stdMat('#ffd9e2', { roughness: 0.95 }), 0, 0.02, 0.4, true);
    add(g, Cy(3.0, 3.0, 0.035, 48), stdMat('#ffeff3', { roughness: 0.95 }), 0, 0.025, 0.4, true);
    add(g, Cy(2.6, 2.6, 0.04, 48), stdMat('#cfe9ff', { roughness: 0.95 }), 0, 0.03, 0.4, true);
    // tường
    var wallBack = add(g, B(14.6, 6.8, 0.3), stdMat('#bcdcff'), 0, 3.4, -6.15); wallBack.castShadow = false;
    add(g, B(14.6, 1.3, 0.34), stdMat('#fff0dd'), 0, 0.65, -6.1, true);
    add(g, B(14.6, 0.12, 0.38), stdMat('#ffffff'), 0, 1.32, -6.08, true);
    var wallL = add(g, B(0.3, 6.8, 13.4), stdMat('#bfeed7'), -7.15, 3.4, 0.6); wallL.castShadow = false;
    add(g, B(0.34, 1.3, 13.4), stdMat('#fff0dd'), -7.1, 0.65, 0.6, true);
    var wallR = add(g, B(0.3, 6.8, 13.4), stdMat('#ffd6bd'), 7.15, 3.4, 0.6); wallR.castShadow = false;
    add(g, B(0.34, 1.3, 13.4), stdMat('#fff0dd'), 7.1, 0.65, 0.6, true);
    // cửa sổ (tường trái + phải)
    var skyTex = canvasTex(256, 256, function (ctx) { var gr = ctx.createLinearGradient(0, 0, 0, 256); gr.addColorStop(0, '#7cc8ff'); gr.addColorStop(1, '#f3fbff'); ctx.fillStyle = gr; ctx.fillRect(0, 0, 256, 256); ctx.fillStyle = 'rgba(255,255,255,.95)'; [[60, 80, 36], [100, 72, 44], [150, 88, 34], [190, 180, 30], [220, 170, 24]].forEach(function (c) { ctx.beginPath(); ctx.arc(c[0], c[1], c[2], 0, 7); ctx.fill(); }); ctx.fillStyle = '#9fd98b'; ctx.fillRect(0, 205, 256, 51); });
    function windowAt(x, z, side) {
      var wg = new T.Group(); wg.position.set(x, 3.3, z); wg.rotation.y = side * Math.PI / 2; g.add(wg);
      add(wg, B(2.6, 2.4, 0.1), stdMat('#ffffff'), 0, 0, 0, true);
      var pane = new T.Mesh(new T.PlaneGeometry(2.3, 2.1), new T.MeshBasicMaterial({ map: skyTex.tex })); pane.position.z = 0.07; wg.add(pane);
      add(wg, B(0.08, 2.1, 0.06), stdMat('#ffffff'), 0, 0, 0.1, true); add(wg, B(2.3, 0.08, 0.06), stdMat('#ffffff'), 0, 0, 0.1, true);
      add(wg, B(2.8, 0.1, 0.3), stdMat('#ffffff'), 0, -1.25, 0.12, true);
    }
    windowAt(-6.95, -2.4, 1); windowAt(-6.95, 3.4, 1); windowAt(6.95, -2.4, -1);
    // cửa ra vào (tường phải)
    var door = new T.Group(); door.position.set(6.95, 0, 4.7); door.rotation.y = -Math.PI / 2; g.add(door);
    add(door, B(2.0, 3.6, 0.12), stdMat('#fff6cf'), 0, 1.8, 0, true);
    add(door, B(1.6, 3.3, 0.05), stdMat('#b8e4ff', { emissive: '#8fd0ff', emissiveIntensity: 0.4 }), 0, 1.7, 0.07, true);
    add(door, B(2.2, 0.16, 0.22), stdMat('#ffb347'), 0, 3.65, 0.05, true);
    var dsign = canvasTex(256, 64, function (ctx) { ctx.fillStyle = '#ffb347'; ctx.fillRect(0, 0, 256, 64); ctx.fillStyle = '#fff'; ctx.font = 'bold 40px ' + FONT; ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillText('LỐI VÀO', 128, 34); });
    var dm = new T.Mesh(new T.PlaneGeometry(1.6, 0.4), new T.MeshBasicMaterial({ map: dsign.tex })); dm.position.set(0, 3.9, 0.17); door.add(dm);
    // bảng đồ thị tri thức
    var board = new T.Group(); board.position.set(0, 4.4, -5.95); g.add(board); clinic.userData.board = board;
    add(board, B(10.2, 4.3, 0.18), stdMat('#f0a04b'), 0, 0, 0, true);
    boardState = canvasTex(2048, 860);
    var bm = new T.Mesh(new T.PlaneGeometry(9.8, 4.11), new T.MeshBasicMaterial({ map: boardState.tex })); bm.position.z = 0.1; board.add(bm);
    drawBoard();
    // đồng hồ + áp phích
    var clk = canvasTex(256, 256, function (ctx) { ctx.fillStyle = '#fff'; ctx.beginPath(); ctx.arc(128, 128, 120, 0, 7); ctx.fill(); ctx.lineWidth = 12; ctx.strokeStyle = '#e8508a'; ctx.stroke(); ctx.strokeStyle = '#16213e'; ctx.lineWidth = 10; ctx.lineCap = 'round'; ctx.beginPath(); ctx.moveTo(128, 128); ctx.lineTo(128, 56); ctx.moveTo(128, 128); ctx.lineTo(176, 150); ctx.stroke(); for (var i = 0; i < 12; i++) { var a = i * Math.PI / 6; ctx.lineWidth = 6; ctx.beginPath(); ctx.moveTo(128 + Math.cos(a) * 98, 128 + Math.sin(a) * 98); ctx.lineTo(128 + Math.cos(a) * 110, 128 + Math.sin(a) * 110); ctx.stroke(); } });
    var cm = new T.Mesh(new T.CircleGeometry(0.55, 32), new T.MeshBasicMaterial({ map: clk.tex })); cm.position.set(-5.4, 5.4, -5.96); g.add(cm);
    var poster = canvasTex(256, 360, function (ctx) { ctx.fillStyle = '#fff'; ctx.fillRect(0, 0, 256, 360); ctx.fillStyle = '#2589e0'; ctx.fillRect(0, 0, 256, 70); ctx.fillStyle = '#fff'; ctx.font = 'bold 34px ' + FONT; ctx.textAlign = 'center'; ctx.fillText('SỐNG KHỎE', 128, 48); ctx.fillStyle = '#e5342d'; ctx.beginPath(); ctx.moveTo(128, 250); ctx.bezierCurveTo(20, 170, 60, 100, 128, 160); ctx.bezierCurveTo(196, 100, 236, 170, 128, 250); ctx.fill(); ctx.fillStyle = '#16213e'; ctx.font = 'bold 28px ' + FONT; ctx.fillText('Ăn rau mỗi ngày', 128, 300); ctx.fillText('Đi bộ 30 phút', 128, 336); });
    var pm = new T.Mesh(new T.PlaneGeometry(1.4, 1.97), new T.MeshBasicMaterial({ map: poster.tex })); pm.position.set(5.7, 4.2, -5.96); g.add(pm);
    add(g, B(1.5, 2.07, 0.06), stdMat('#ffb347'), 5.7, 4.2, -5.99, true);
    // đèn treo
    [-4.8, 4.8].forEach(function (x) {
      add(g, Cy(0.015, 0.015, 1.2, 6), stdMat('#555'), x, 6.3, -0.5, true);
      add(g, Cy(0.2, 0.8, 0.5, 20), stdMat(x < 0 ? '#ffd166' : '#ffb3c7'), x, 5.5, -0.5, true);
      add(g, Sp(0.2, 12, 10), new T.MeshBasicMaterial({ color: '#fffbe0' }), x, 5.2, -0.5, true);
    });
    // quầy tiếp tân
    var rc = new T.Group(); rc.position.set(-2.6, 0, 5.4); g.add(rc);
    add(rc, B(3.2, 1.05, 0.9), stdMat('#fff6e3'), 0, 0.52, 0); add(rc, B(3.4, 0.1, 1.1), stdMat('#ff9f68'), 0, 1.08, 0.02);
    add(rc, B(3.2, 0.2, 0.92), stdMat('#ffb347'), 0, 0.15, 0.01, true);
    var rs = makeSign('Tiếp tân', 'Xin mời đăng ký', '#ff8a4c'); rs.position.set(0, 0.62, 0.46); rc.add(rs);
    add(rc, B(0.5, 0.35, 0.05), stdMat('#25304a'), -0.9, 1.32, -0.1); add(rc, B(0.12, 0.2, 0.1), stdMat('#25304a'), -0.9, 1.15, -0.1);
    add(rc, Cy(0.12, 0.1, 0.18, 12), stdMat('#ffd166'), 1.1, 1.2, 0); add(rc, Sp(0.1, 10, 8), stdMat('#ff6b6b'), 1.1, 1.36, 0);
    // giường khám (tường trái)
    var bed = new T.Group(); bed.position.set(-6.0, 0, 0.2); g.add(bed);
    add(bed, B(1.5, 0.5, 3.2), stdMat('#ffffff'), 0, 0.5, 0); add(bed, B(1.6, 0.18, 3.3), stdMat('#7fd6f5'), 0, 0.82, 0);
    add(bed, B(1.4, 0.16, 3.0), stdMat('#fffaf0'), 0, 0.98, 0.1); add(bed, B(1.2, 0.18, 0.7), stdMat('#ffffff'), 0, 1.1, -1.25);
    [-1, 1].forEach(function (a) { [-1, 1].forEach(function (b) { add(bed, Cy(0.07, 0.07, 0.4, 8), stdMat('#9aa5b8'), a * 0.7, 0.2, b * 1.5); }); });
    // cây
    function plant(x, z, s) { var p = new T.Group(); p.position.set(x, 0, z); p.scale.setScalar(s); g.add(p); add(p, Cy(0.38, 0.28, 0.6, 14), stdMat('#ff9f68'), 0, 0.3, 0); [[0, 1.0, 0, 0.5], [0.3, 1.3, 0.1, 0.4], [-0.3, 1.25, -0.1, 0.42], [0.05, 1.6, 0.05, 0.32]].forEach(function (b) { add(p, Sp(b[3], 14, 10), stdMat(['#4caf50', '#66bb6a', '#43a047', '#7bd17f'][Math.floor(b[1] * 3) % 4]), b[0], b[1], b[2]); }); }
    plant(-6.4, 5.4, 1); plant(6.3, 6.0, 1.15); plant(2.9, -4.7, 0.85); plant(-6.4, -5.2, 0.9);
    // tủ + máy lọc nước cạnh bàn
    var cab = add(g, B(1.2, 1.5, 0.7), stdMat('#ffffff'), 3.4, 0.75, -5.5); add(g, B(1.0, 0.04, 0.02), stdMat('#cfe0ee'), 3.4, 1.1, -5.14, true);
    add(g, Cy(0.16, 0.16, 0.3, 10), stdMat('#7fd6f5', { transparent: true, opacity: 0.8 }), 3.4, 1.7, -5.5);
  }

  /* bàn làm việc cho từng nhân vật */
  function buildDesk(id, def) {
    var g = new T.Group(), p = chars[id].position;
    var dir = new T.Vector3(FACE_POINT.x - p.x, 0, FACE_POINT.z - p.z).normalize();
    g.position.set(p.x + dir.x * 1.2, 0, p.z + dir.z * 1.2);
    g.rotation.y = Math.atan2(dir.x, dir.z);
    var c = def.color;
    add(g, B(1.9, 0.78, 0.9), stdMat(shade(c, 0.32)), 0, 0.39, 0);
    add(g, B(2.0, 0.1, 1.04), stdMat('#ffffff'), 0, 0.82, 0);
    add(g, B(2.0, 0.12, 0.92), stdMat(c), 0, 0.06, 0, true);
    var sign = makeSign(def.name, def.role, c); sign.position.set(0, 0.42, 0.455); g.add(sign); chars[id].userData.sign = sign;
    var k = def.kind;
    if (k === 'doctor') {
      add(g, B(0.5, 0.04, 0.34), stdMat('#fffaf0'), -0.5, 0.88, 0.05); add(g, Cy(0.05, 0.05, 0.14, 8), stdMat('#e5342d'), 0.6, 0.94, 0);
      var tor = add(g, new T.TorusGeometry(0.16, 0.025, 8, 18), stdMat('#25304a'), 0.2, 0.92, 0.1); tor.rotation.x = Math.PI / 2;
    } else if (k === 'predictor') {
      add(g, B(0.9, 0.55, 0.06), stdMat('#25304a'), 0, 1.22, -0.12).rotation.x = -0.1; add(g, B(0.16, 0.25, 0.1), stdMat('#25304a'), 0, 0.98, -0.16);
      var sc = canvasTex(256, 140, function (ctx) { ctx.fillStyle = '#0f2748'; ctx.fillRect(0, 0, 256, 140); [['#7fe3ff', 90], ['#ffe27f', 55], ['#ff9db5', 100], ['#a0f0c0', 70]].forEach(function (b, i) { ctx.fillStyle = b[0]; ctx.fillRect(24 + i * 56, 130 - b[1], 36, b[1]); }); });
      var sm = new T.Mesh(new T.PlaneGeometry(0.82, 0.45), new T.MeshBasicMaterial({ map: sc.tex })); sm.position.set(0, 1.22, -0.085); sm.rotation.x = -0.1; g.add(sm);
    } else if (k === 'knowledge') {
      [['#ff7a59', 0.5], ['#ffd166', 0.42], ['#5ec8a0', 0.55], ['#7da7ff', 0.38]].forEach(function (b, i) { add(g, B(b[1], 0.1, 0.34), stdMat(b[0]), -0.55 + i * 0.02, 0.92 + i * 0.1, 0.0).rotation.y = i * 0.15; });
      add(g, Cy(0.04, 0.04, 0.5, 8), stdMat('#25304a'), 0.6, 1.1, -0.1); add(g, Sp(0.16, 12, 10), stdMat('#fff2a8', { emissive: '#ffd84a', emissiveIntensity: 0.6 }), 0.6, 1.4, -0.1);
    } else if (k === 'coach') {
      [-0.45, -0.1].forEach(function (x, i) { var d = new T.Group(); d.position.set(x, 0.95, 0.05); d.rotation.y = i; g.add(d); add(d, Cy(0.03, 0.03, 0.4, 8), stdMat('#25304a'), 0, 0, 0).rotation.z = Math.PI / 2; [-1, 1].forEach(function (s) { add(d, Cy(0.1, 0.1, 0.08, 12), stdMat(i ? '#ffd166' : '#ff7a59'), s * 0.2, 0, 0).rotation.z = Math.PI / 2; }); });
      add(g, Cy(0.09, 0.09, 0.3, 12), stdMat('#7fd6f5'), 0.6, 1.0, 0);
    } else if (k === 'guardian') {
      add(g, Cy(0.2, 0.26, 0.16, 16), stdMat('#25304a'), 0.55, 0.95, 0);
      siren = add(g, Sp(0.21, 16, 12, 0, Math.PI * 2, 0, Math.PI / 2), new T.MeshStandardMaterial({ color: '#ffb199', emissive: '#e5342d', emissiveIntensity: 0.1, transparent: true, opacity: 0.95 }), 0.55, 1.03, 0);
      sirenLight = new T.PointLight('#ff3b30', 0, 7, 2); sirenLight.position.set(0.55, 1.5, 0.2); g.add(sirenLight);
      add(g, B(0.5, 0.04, 0.34), stdMat('#fffaf0'), -0.5, 0.88, 0.05); add(g, B(0.12, 0.12, 0.02), stdMat('#e5342d'), -0.5, 0.91, 0.05);
    } else {
      add(g, B(0.7, 0.3, 0.5), stdMat('#ffffff'), -0.35, 1.0, -0.05); add(g, B(0.5, 0.04, 0.4), stdMat('#fffaf0'), -0.35, 1.19, 0.12).rotation.x = -0.3;
      add(g, B(0.4, 0.2, 0.3), stdMat('#fffaf0'), 0.5, 0.95, 0.05);
    }
    clinic.add(g); return g;
  }

  /* màn hình theo dõi đường huyết */
  function makeMonitor(w, h, standing) {
    var g = new T.Group();
    var cvs = canvasTex(1024, Math.round(1024 * h / w));
    var mon = { ct: cvs, w: 1024, h: Math.round(1024 * h / w), group: g };
    var frame = add(g, B(w + 0.18, h + 0.18, 0.12), stdMat('#25304a'), 0, 0, 0);
    var screen = new T.Mesh(new T.PlaneGeometry(w, h), new T.MeshBasicMaterial({ map: cvs.tex })); screen.position.z = 0.07; g.add(screen);
    if (standing) { add(g, Cy(0.05, 0.05, 1.4, 10), stdMat('#9aa5b8'), 0, -h / 2 - 0.7, -0.02); add(g, Cy(0.45, 0.5, 0.08, 20), stdMat('#25304a'), 0, -h / 2 - 1.4, -0.02); }
    monitors.push(mon); drawMonitor(mon); return g;
  }
  function drawMonitor(mon) {
    var ctx = mon.ct.ctx, W2 = mon.w, H2 = mon.h, z = ZONES[glucoseState.zone] || ZONES.ok;
    ctx.fillStyle = '#0c1f3d'; ctx.fillRect(0, 0, W2, H2);
    var gx = W2 * 0.05, gy = H2 * 0.6, gw = W2 * 0.9, gh = H2 * 0.34;
    // dải an toàn 70-180, thang 40-300
    function yOf(v) { return gy + gh - (clamp(v, 40, 300) - 40) / 260 * gh; }
    ctx.fillStyle = 'rgba(34,197,94,0.28)'; ctx.fillRect(gx, yOf(180), gw, yOf(70) - yOf(180));
    ctx.strokeStyle = 'rgba(255,255,255,0.35)'; ctx.lineWidth = 2; ctx.setLineDash([8, 8]);
    [70, 180].forEach(function (v) { ctx.beginPath(); ctx.moveTo(gx, yOf(v)); ctx.lineTo(gx + gw, yOf(v)); ctx.stroke(); }); ctx.setLineDash([]);
    ctx.fillStyle = '#9fe9b5'; ctx.font = 'bold ' + Math.round(H2 * 0.05) + 'px ' + FONT; ctx.textBaseline = 'middle'; ctx.fillText('70', gx + 4, yOf(70) - 14); ctx.fillText('180', gx + 4, yOf(180) + 14);
    var s = glucoseState.series;
    if (s.length > 1) {
      ctx.lineWidth = Math.max(5, H2 * 0.016); ctx.lineJoin = 'round'; ctx.strokeStyle = '#7fe3ff'; ctx.beginPath();
      s.forEach(function (p, i) { var x = gx + gw * (p.hour - 6) / 24, y = yOf(p.glucose); if (i) ctx.lineTo(x, y); else ctx.moveTo(x, y); }); ctx.stroke();
      var last = s[s.length - 1]; ctx.fillStyle = z.color; ctx.beginPath(); ctx.arc(gx + gw * (last.hour - 6) / 24, yOf(last.glucose), H2 * 0.03, 0, 7); ctx.fill();
    }
    ctx.fillStyle = z.color; ctx.font = 'bold ' + Math.round(H2 * 0.3) + 'px ' + FONT; ctx.textBaseline = 'alphabetic';
    var vt = String(Math.round(glucoseState.value)); ctx.fillText(vt, gx, H2 * 0.56); var vw = ctx.measureText(vt).width;
    ctx.fillStyle = '#e6f0ff'; ctx.font = 'bold ' + Math.round(H2 * 0.075) + 'px ' + FONT; ctx.fillText('mg/dL', gx + vw + 12, H2 * 0.56);
    var hh = Math.floor(glucoseState.hour) % 24;
    ctx.fillStyle = '#ffd166'; ctx.font = 'bold ' + Math.round(H2 * 0.1) + 'px ' + FONT; ctx.fillText('Giờ ' + (hh < 10 ? '0' : '') + hh + ':00', gx, H2 * 0.13);
    ctx.fillStyle = z.color; rrect(ctx, W2 * 0.62, H2 * 0.03, W2 * 0.33, H2 * 0.14, 14); ctx.fill();
    ctx.fillStyle = '#ffffff'; ctx.font = 'bold ' + Math.round(H2 * 0.09) + 'px ' + FONT; ctx.textAlign = 'center'; ctx.fillText(z.label.toUpperCase(), W2 * 0.785, H2 * 0.14); ctx.textAlign = 'left';
    mon.ct.tex.needsUpdate = true;
  }

  /* ---------- bảng đồ thị trên tường ---------- */
  function drawBoard() {
    var ctx = boardState.ctx, W2 = 2048, H2 = 860;
    var bg = ctx.createLinearGradient(0, 0, W2, H2); bg.addColorStop(0, '#ffffff'); bg.addColorStop(1, '#eaf4ff');
    ctx.fillStyle = bg; ctx.fillRect(0, 0, W2, H2);
    ctx.fillStyle = '#16213e'; ctx.font = 'bold 64px ' + FONT; ctx.textBaseline = 'middle'; ctx.fillText('Đồ thị tri thức', 40, 54);
    // chú giải
    var lx = W2 - 40;
    Object.keys(TYPE_COLORS).reverse().forEach(function (k) { ctx.font = 'bold 34px ' + FONT; var tw = ctx.measureText(TYPE_LABELS[k]).width; lx -= tw; ctx.fillStyle = '#16213e'; ctx.fillText(TYPE_LABELS[k], lx, 54); lx -= 26; ctx.fillStyle = TYPE_COLORS[k]; ctx.beginPath(); ctx.arc(lx, 54, 12, 0, 7); ctx.fill(); lx -= 26; });
    if (!kgData || !kgLayout2d) { ctx.fillStyle = '#5b6b8a'; ctx.font = 'bold 48px ' + FONT; ctx.textAlign = 'center'; ctx.fillText('Đang tải đồ thị tri thức...', W2 / 2, H2 / 2 + 30); ctx.textAlign = 'left'; boardState.tex.needsUpdate = true; return; }
    var hl = kgHighlight; var X = function (p) { return 190 + p[0] * (W2 - 380); }, Y = function (p) { return 150 + p[1] * (H2 - 230); };
    kgData.edges.forEach(function (e) {
      var a = kgLayout2d[e.source], b = kgLayout2d[e.target]; if (!a || !b) return;
      var on = hl && hl[e.source] && hl[e.target];
      ctx.strokeStyle = on ? '#ff8a00' : 'rgba(70,90,130,0.55)'; ctx.lineWidth = on ? 7 : 3; ctx.beginPath(); ctx.moveTo(X(a), Y(a)); ctx.lineTo(X(b), Y(b)); ctx.stroke();
    });
    kgData.nodes.forEach(function (n) {
      var p = kgLayout2d[n.id]; if (!p) return; var on = hl && hl[n.id], dim = hl && !on;
      var x = X(p), y = Y(p), r = on ? 30 : 20;
      if (on) { ctx.fillStyle = 'rgba(255,170,0,0.35)'; ctx.beginPath(); ctx.arc(x, y, 54, 0, 7); ctx.fill(); ctx.fillStyle = 'rgba(255,170,0,0.55)'; ctx.beginPath(); ctx.arc(x, y, 42, 0, 7); ctx.fill(); }
      ctx.globalAlpha = dim ? 0.45 : 1; ctx.fillStyle = TYPE_COLORS[n.type] || '#667'; ctx.beginPath(); ctx.arc(x, y, r, 0, 7); ctx.fill();
      ctx.lineWidth = 5; ctx.strokeStyle = '#fff'; ctx.stroke();
      ctx.font = 'bold ' + (on ? 42 : 33) + 'px ' + FONT; ctx.textAlign = 'center'; ctx.lineWidth = 8; ctx.strokeStyle = '#ffffff'; ctx.strokeText(n.vi || n.label, x, y - r - 16);
      ctx.fillStyle = on ? '#b45300' : '#16213e'; ctx.fillText(n.vi || n.label, x, y - r - 16); ctx.textAlign = 'left'; ctx.globalAlpha = 1;
    });
    boardState.tex.needsUpdate = true;
  }

  /* ---------- bố cục đồ thị (lực) ---------- */
  function forceLayout(nodes, edges, dim, seed) {
    var rnd = mulberry(seed || 7), n = nodes.length, idx = {}, P = [], V = [];
    var types = Object.keys(TYPE_COLORS);
    nodes.forEach(function (nd, i) {
      idx[nd.id] = i; var ti = Math.max(0, types.indexOf(nd.type));
      if (dim === 3) {
        var y = 1 - (ti + 0.5) / types.length * 2, rr = Math.sqrt(1 - y * y), th = ti * 2.399963;
        P.push([Math.cos(th) * rr * 7 + (rnd() - 0.5) * 3.6, y * 7 + (rnd() - 0.5) * 3.6, Math.sin(th) * rr * 7 + (rnd() - 0.5) * 3.6]);
      } else { var a = ti / types.length * Math.PI * 2 + rnd() * 0.5; P.push([Math.cos(a) * 2 + rnd() - 0.5, Math.sin(a) * 2 + rnd() - 0.5, 0]); }
      V.push([0, 0, 0]);
    });
    var L = dim === 3 ? 3.4 : 2.0, rep = dim === 3 ? 26 : 6;
    for (var it = 0; it < 420; it++) {
      var cool = 1 - it / 420;
      for (var i = 0; i < n; i++) for (var j = i + 1; j < n; j++) {
        var dx = P[i][0] - P[j][0], dy = P[i][1] - P[j][1], dz = P[i][2] - P[j][2], d2 = dx * dx + dy * dy + dz * dz + 0.05, d = Math.sqrt(d2), f = rep / d2;
        V[i][0] += dx / d * f; V[i][1] += dy / d * f; V[i][2] += dz / d * f; V[j][0] -= dx / d * f; V[j][1] -= dy / d * f; V[j][2] -= dz / d * f;
      }
      edges.forEach(function (e) {
        var a = idx[e.source], b = idx[e.target]; if (a == null || b == null) return;
        var dx = P[b][0] - P[a][0], dy = P[b][1] - P[a][1], dz = P[b][2] - P[a][2], d = Math.sqrt(dx * dx + dy * dy + dz * dz) + 0.01, f = (d - L) * (dim === 3 ? 0.03 : 0.08);
        V[a][0] += dx / d * f; V[a][1] += dy / d * f; V[a][2] += dz / d * f; V[b][0] -= dx / d * f; V[b][1] -= dy / d * f; V[b][2] -= dz / d * f;
      });
      for (var k = 0; k < n; k++) {
        for (var c = 0; c < 3; c++) { V[k][c] -= P[k][c] * 0.006; P[k][c] += clamp(V[k][c], -0.4, 0.4) * cool * 0.5; V[k][c] *= 0.6; }
        if (dim !== 3) P[k][2] = 0;
      }
    }
    return { idx: idx, P: P };
  }

  /* ---------- đồ thị 3D ---------- */
  function textSprite(text, opt) {
    opt = opt || {};
    var fs = opt.size || 40, c = document.createElement('canvas'), ctx = c.getContext('2d');
    ctx.font = 'bold ' + fs + 'px ' + FONT; var tw = Math.ceil(ctx.measureText(text).width) + 36; c.width = tw; c.height = fs + 28;
    ctx = c.getContext('2d'); ctx.font = 'bold ' + fs + 'px ' + FONT; ctx.textBaseline = 'middle';
    ctx.fillStyle = opt.bg || 'rgba(255,255,255,0.96)'; rrect(ctx, 3, 3, c.width - 6, c.height - 6, c.height / 2.4); ctx.fill();
    ctx.lineWidth = 4; ctx.strokeStyle = opt.border || '#16213e'; ctx.stroke();
    ctx.fillStyle = opt.color || '#16213e'; ctx.fillText(text, 18, c.height / 2 + 2);
    var tex = new T.CanvasTexture(c); tex.minFilter = T.LinearFilter;
    var sp = new T.Sprite(new T.SpriteMaterial({ map: tex, transparent: true, depthTest: opt.depthTest !== false })); var sc = (opt.world || 0.02);
    sp.scale.set(c.width * sc, c.height * sc, 1); return sp;
  }

  function buildKG(data) {
    if (kgGroup) { scene.remove(kgGroup); }
    kgGroup = new T.Group(); kgGroup.visible = (mode === 'kg'); scene.add(kgGroup);
    kgNodeMeshes = []; kgEdgeLabels = [];
    var lay = forceLayout(data.nodes, data.edges, 3, 11), scale = 1.0, maxR = 0.01;
    lay.P.forEach(function (p) { maxR = Math.max(maxR, Math.hypot(p[0], p[1], p[2])); }); scale = 9 / maxR; kgRadius = 9;
    var deg = {}; data.edges.forEach(function (e) { deg[e.source] = (deg[e.source] || 0) + 1; deg[e.target] = (deg[e.target] || 0) + 1; });
    var posById = {};
    data.nodes.forEach(function (n, i) {
      var p = lay.P[i], v = new T.Vector3(p[0] * scale, p[1] * scale, p[2] * scale), col = TYPE_COLORS[n.type] || '#667';
      posById[n.id] = v;
      var r = 0.3 + Math.min(deg[n.id] || 0, 9) * 0.05;
      var m = new T.Mesh(Sp(r, 24, 18), stdMat(col, { roughness: 0.35, emissive: col, emissiveIntensity: 0.12 })); m.position.copy(v); m.userData = { id: n.id, r: r, node: n, baseScale: 1 };
      kgGroup.add(m); kgNodeMeshes.push(m);
      var lab = textSprite(n.vi || n.label, { size: 44, border: col, world: 0.0058, color: '#10192f' }); lab.position.copy(v).add(new T.Vector3(0, r + 0.36, 0)); lab.renderOrder = 5; m.userData.label = lab; kgGroup.add(lab);
      var halo = new T.Mesh(Sp(r * 1.45, 20, 14), new T.MeshBasicMaterial({ color: '#ffb300', transparent: true, opacity: 0.0, depthWrite: false })); halo.position.copy(v); kgGroup.add(halo); m.userData.halo = halo;
    });
    data.edges.forEach(function (e) {
      var a = posById[e.source], b = posById[e.target]; if (!a || !b) return;
      var dir = new T.Vector3().subVectors(b, a), len = dir.length(); var mid = new T.Vector3().addVectors(a, b).multiplyScalar(0.5);
      var cyl = new T.Mesh(Cy(0.04, 0.04, len, 6), new T.MeshBasicMaterial({ color: '#5b6f99' })); cyl.position.copy(mid);
      cyl.quaternion.setFromUnitVectors(new T.Vector3(0, 1, 0), dir.clone().normalize()); kgGroup.add(cyl); e._mesh = cyl;
      var tr = 0.3 + Math.min(deg[e.target] || 0, 9) * 0.05, cone = new T.Mesh(new T.ConeGeometry(0.12, 0.3, 10), new T.MeshBasicMaterial({ color: '#33456e' }));
      cone.position.copy(b).addScaledVector(dir.clone().normalize(), -(tr + 0.16)); cone.quaternion.setFromUnitVectors(new T.Vector3(0, 1, 0), dir.clone().normalize()); kgGroup.add(cone);
      var lbl = textSprite(relVi(e.rel), { size: 28, bg: 'rgba(255,248,225,0.97)', border: '#e0a43a', world: 0.0048, depthTest: false }); lbl.position.copy(mid); lbl.renderOrder = 4; lbl.userData.src = e.source; lbl.userData.dst = e.target; kgGroup.add(lbl); kgEdgeLabels.push(lbl);
    });
  }

  function applyKgHighlight() {
    kgNodeMeshes.forEach(function (m) {
      var on = kgHighlight && kgHighlight[m.userData.id], sel = kgSelected === m.userData.id, dim = kgHighlight && !on;
      m.material.transparent = !!dim; m.material.opacity = dim ? 0.3 : 1; m.material.emissiveIntensity = on ? 0.7 : 0.12;
      m.userData.baseScale = (on || sel) ? 1.55 : 1; m.userData.label.material.opacity = dim ? 0.35 : 1;
      m.userData.halo.material.opacity = (on || sel) ? 0.35 : 0;
    });
    kgEdgeLabels.forEach(function (l) { var on = kgHighlight && kgHighlight[l.userData.src] && kgHighlight[l.userData.dst]; l.material.opacity = kgHighlight ? (on ? 1 : 0.28) : 1; });
    kgData && kgData.edges.forEach(function (e) { if (e._mesh) e._mesh.material.color.set(kgHighlight && kgHighlight[e.source] && kgHighlight[e.target] ? '#ff8a00' : '#5b6f99'); });
  }

  /* ---------- camera ---------- */
  function homeView() {
    var asp = W / Math.max(H, 1), d = 13.6 * Math.max(1, 1.5 / asp);
    if (mode === 'kg') return { pos: new T.Vector3(0, 3, Math.max(12, kgRadius * 2.0) * Math.max(1, 1.5 / asp)), tgt: new T.Vector3(0, 0, 0) };
    var tgt = new T.Vector3(0, 1.8, -0.4), pol = 1.28;
    return { pos: new T.Vector3(0, tgt.y + d * Math.cos(pol), tgt.z + d * Math.sin(pol)), tgt: tgt };
  }
  function flyTo(pos, tgt, ms) {
    tween = { t0: performance.now(), dur: ms || 1300, p0: camera.position.clone(), t0v: controls.target.clone(), p1: pos, t1: tgt };
  }
  function flyToChar(id, ms) {
    var c = id === 'patient' ? patient : chars[id]; if (!c) return;
    var asp = W / Math.max(H, 1), d = 7.4 * Math.max(1, 1.5 / asp), az = clamp(c.rotation.y * 0.7, -0.7, 0.7), pol = 1.33;
    var tgt = new T.Vector3(c.position.x, 1.9, c.position.z - 0.2);
    var pos = new T.Vector3(tgt.x + Math.sin(az) * Math.sin(pol) * d, tgt.y + Math.cos(pol) * d, tgt.z + Math.cos(az) * Math.sin(pol) * d);
    flyTo(pos, tgt, ms);
  }
  function flyToBoard(ms) {
    var asp = W / Math.max(H, 1), d = 11.5 * Math.max(1, 1.5 / asp);
    var tgt = new T.Vector3(1.0, 3.2, -4.5), pol = 1.35;
    flyTo(new T.Vector3(tgt.x, tgt.y + d * Math.cos(pol), tgt.z + d * Math.sin(pol)), tgt, ms);
  }
  function flyHome(ms) { var h = homeView(); flyTo(h.pos, h.tgt, ms); }
  function flyLife(ms) {
    var asp = W / Math.max(H, 1), d = 11 * Math.max(1, 1.5 / asp), tgt = new T.Vector3(0.0, 1.9, 0.0), pol = 1.3;
    flyTo(new T.Vector3(tgt.x, tgt.y + d * Math.cos(pol), tgt.z + d * Math.sin(pol)), tgt, ms);
  }

  function applyControlLimits() {
    if (mode === 'kg') {
      controls.minAzimuthAngle = -Infinity; controls.maxAzimuthAngle = Infinity; controls.minPolarAngle = 0.15; controls.maxPolarAngle = Math.PI - 0.15;
      controls.minDistance = 4; controls.maxDistance = 40; controls.autoRotate = true; controls.autoRotateSpeed = 0.7;
    } else {
      controls.minAzimuthAngle = -1.0; controls.maxAzimuthAngle = 1.0; controls.minPolarAngle = 0.62; controls.maxPolarAngle = 1.5;
      controls.minDistance = 3.5; controls.maxDistance = 26; controls.autoRotate = false;
    }
  }

  /* ---------- bong bóng thoại (HTML) ---------- */
  var bubbles = {};
  function getBubble(id) {
    if (bubbles[id]) return bubbles[id];
    var el = document.createElement('div'); el.className = 'bubble'; el.style.display = 'none';
    el.innerHTML = '<div class="bubble-name"></div><div class="bubble-text"></div>'; bubblesEl.appendChild(el);
    bubbles[id] = { el: el, nameEl: el.firstChild, textEl: el.lastChild, timer: null, lift: 0 }; return bubbles[id];
  }
  function charInfo(id) {
    if (id === 'patient') return { name: patient.userData.displayName || 'Bệnh nhân', color: patient.userData.color || '#2589e0' };
    var c = chars[id]; return c ? { name: c.userData.displayName, color: c.userData.color } : { name: id, color: '#2589e0' };
  }
  function say(id, text, opt) {
    opt = opt || {}; var b = getBubble(id), info = charInfo(id);
    if (b.timer) { clearInterval(b.timer); b.timer = null; }
    b.el.style.display = 'block'; b.el.style.setProperty('--c', info.color); b.nameEl.textContent = info.name; b.el.classList.toggle('sticky', !!opt.sticky);
    b.el.classList.toggle('compact', !!opt.compact); b.el.classList.toggle('mini', !!opt.mini);
    var typeMs = opt.typeMs == null ? 0 : opt.typeMs, i = 0;
    if (!typeMs) { b.textEl.textContent = text; }
    else {
      var step = Math.max(14, typeMs / Math.max(text.length, 1)); b.textEl.textContent = '';
      b.timer = setInterval(function () { i += Math.max(1, Math.round(16 / step)); b.textEl.textContent = text.slice(0, i); if (i >= text.length) { clearInterval(b.timer); b.timer = null; } }, 16 > step ? 16 : step);
    }
    b.lastText = text; b.visible = true; b.followChar = id;
  }
  function hideBubble(id) { var b = bubbles[id]; if (!b) return; if (b.timer) { clearInterval(b.timer); b.timer = null; } b.el.style.display = 'none'; b.visible = false; }
  function hideAllBubbles() { Object.keys(bubbles).forEach(hideBubble); }
  var tmpV = new T.Vector3();
  function updateBubbles() {
    var rect = { w: W, h: H }, placed = [];
    Object.keys(bubbles).forEach(function (id) {
      var b = bubbles[id]; if (!b.visible) return;
      var c = id === 'patient' ? patient : chars[id]; if (!c || mode === 'kg') { b.el.style.visibility = 'hidden'; return; }
      tmpV.set(c.position.x, id === 'patient' ? 3.7 : 4.0, c.position.z); tmpV.project(camera);
      if (tmpV.z > 1) { b.el.style.visibility = 'hidden'; return; }
      var bw = b.el.offsetWidth, bh = b.el.offsetHeight;
      var x = (tmpV.x * 0.5 + 0.5) * W, y = (-tmpV.y * 0.5 + 0.5) * H;
      x = clamp(x, bw / 2 + 8, W - bw / 2 - 8); y = clamp(y, bh + 14, H - 10);
      // tránh chồng nhau: đẩy xuống nếu hết chỗ phía trên, luôn giữ bong bóng trong khung nhìn
      placed.forEach(function (p) { if (Math.abs(p.x - x) < (p.w + bw) / 2 && Math.abs(p.y - y) < (p.h + 8)) { var up = p.y - p.h - 10; y = up >= bh + 14 ? up : p.y + bh + 10; } });
      y = clamp(y, bh + 14, H - 10);
      placed.push({ x: x, y: y, w: bw, h: bh });
      b.el.style.visibility = 'visible'; b.el.style.transform = 'translate(' + Math.round(x - bw / 2) + 'px,' + Math.round(y - bh) + 'px)';
    });
  }

  /* ---------- hiệu ứng tia sáng handoff ---------- */
  function beam(fromId, toId, color, ms) {
    var a = chars[fromId], b = chars[toId]; if (!a || !b) return;
    color = color || a.userData.color; ms = ms || 1000;
    var p0 = new T.Vector3(a.position.x, 2.9, a.position.z), p2 = new T.Vector3(b.position.x, 2.9, b.position.z);
    var p1 = new T.Vector3().addVectors(p0, p2).multiplyScalar(0.5); p1.y += 1.8;
    var curve = new T.QuadraticBezierCurve3(p0, p1, p2);
    var g = new T.Group(); clinic.add(g);
    var tube = new T.Mesh(new T.TubeGeometry(curve, 28, 0.05, 6), new T.MeshBasicMaterial({ color: color, transparent: true, opacity: 0.0 })); g.add(tube);
    var sparks = []; for (var i = 0; i < 9; i++) { var s = new T.Mesh(Sp(0.1 - i * 0.006, 8, 6), new T.MeshBasicMaterial({ color: color, transparent: true })); g.add(s); sparks.push(s); }
    var arrow = new T.Mesh(new T.ConeGeometry(0.2, 0.5, 12), new T.MeshBasicMaterial({ color: color })); arrow.visible = false; g.add(arrow);
    var t0 = performance.now();
    fxList.push({ update: function () {
      var k = (performance.now() - t0) / ms; if (k >= 1.15) { clinic.remove(g); return false; }
      var head = clamp(k / 0.8, 0, 1);
      tube.material.opacity = k < 0.8 ? 0.45 : Math.max(0, 0.45 * (1.15 - k) / 0.35);
      sparks.forEach(function (s, i) { var u = clamp(head - i * 0.03, 0, 1); curve.getPoint(ease(u), s.position); s.visible = u > 0 && u < 1; s.material.opacity = 1 - i * 0.09; });
      if (head >= 1) { arrow.visible = true; arrow.position.copy(p2).add(new T.Vector3(0, 0.55 + Math.sin(k * 20) * 0.08, 0)); arrow.rotation.x = Math.PI; var s = 1 + Math.sin(k * 25) * 0.15; arrow.scale.setScalar(s); }
      return true;
    } });
  }

  /* ---------- vòng lặp ---------- */
  function resize() {
    if (!stage) return; var r = stage.getBoundingClientRect(); W = Math.max(200, Math.round(r.width)); H = Math.max(200, Math.round(r.height));
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2)); renderer.setSize(W, H, false);
    canvas.style.width = W + 'px'; canvas.style.height = H + 'px'; camera.aspect = W / H; camera.updateProjectionMatrix();
  }

  var lastT = 0;
  function frame(now) {
    requestAnimationFrame(frame);
    if (document.hidden) { lastT = 0; return; }
    var t = now / 1000, dt = lastT ? Math.min(0.05, t - lastT) : 0.016; lastT = t;
    if (tween) {
      var k = clamp((performance.now() - tween.t0) / tween.dur, 0, 1), e = ease(k);
      camera.position.lerpVectors(tween.p0, tween.p1, e); controls.target.lerpVectors(tween.t0v, tween.t1, e);
      if (k >= 1) tween = null;
    }
    controls.update();
    if (mode !== 'kg') {
      Object.keys(chars).forEach(function (id) { updateChar(chars[id], t, dt); });
      updatePatient(t, dt); blinkEyes(patient.userData, t, patient.userData.closed);
      patient.userData.mouth.scale.y = patient.userData.speaking ? 0.4 + Math.abs(Math.sin(t * 12)) : 0.45;
      // cảnh báo
      var on = performance.now() < alertState.until;
      var blinkv = on ? (Math.sin(t * 14) > 0 ? 1 : 0.15) : 0;
      if (siren) { siren.material.emissive.set(alertState.color); siren.material.emissiveIntensity = on ? 0.2 + blinkv * 1.4 : 0.1; }
      if (sirenLight) { sirenLight.color.set(alertState.color); sirenLight.intensity = blinkv * 3.5; }
      if (stage) { if (on !== stage._alert) { stage._alert = on; stage.classList.toggle('alerting', on); stage.style.setProperty('--alert', alertState.color); } }
      fxList = fxList.filter(function (f) { return f.update(dt); });
    } else {
      kgNodeMeshes.forEach(function (m) {
        var s = m.userData.baseScale * (kgHighlight && kgHighlight[m.userData.id] ? 1 + Math.sin(t * 6) * 0.1 : 1) * (hoverId === m.userData.id ? 1.2 : 1);
        m.scale.setScalar(lerp(m.scale.x, s, 0.2)); m.userData.halo.scale.setScalar(m.scale.x);
      });
    }
    renderer.render(scene, camera);
    updateBubbles();
  }

  /* ---------- KG: chọn nút ---------- */
  function pickNode(ev) {
    var r = canvas.getBoundingClientRect(); mouse.x = ((ev.clientX - r.left) / r.width) * 2 - 1; mouse.y = -((ev.clientY - r.top) / r.height) * 2 + 1;
    raycaster.setFromCamera(mouse, camera); var hits = raycaster.intersectObjects(kgNodeMeshes, false); return hits.length ? hits[0].object.userData.id : null;
  }

  /* ---------- API công khai ---------- */
  var Clinic = {
    ZONES: ZONES, TYPE_COLORS: TYPE_COLORS, TYPE_LABELS: TYPE_LABELS, relVi: relVi,
    init: function (stageEl, bubblesContainer) {
      stage = stageEl; bubblesEl = bubblesContainer;
      canvas = document.createElement('canvas'); canvas.className = 'gl'; stage.insertBefore(canvas, stage.firstChild);
      renderer = new T.WebGLRenderer({ canvas: canvas, antialias: true });
      renderer.shadowMap.enabled = true; renderer.shadowMap.type = T.PCFSoftShadowMap; renderer.setClearColor('#cfe8ff');
      scene = new T.Scene(); scene.background = new T.Color('#cfe8ff');
      camera = new T.PerspectiveCamera(42, 1.5, 0.1, 120);
      controls = new T.OrbitControls(camera, canvas); controls.enablePan = false; controls.enableDamping = true; controls.dampingFactor = 0.08; controls.rotateSpeed = 0.6;
      controls.addEventListener('start', function () { tween = null; });
      hemi = new T.HemisphereLight('#ffffff', '#ffe4c4', 0.55); scene.add(hemi);
      scene.add(new T.AmbientLight('#ffffff', 0.2));
      var sun = new T.DirectionalLight('#fff4e0', 0.6); sun.position.set(6, 13, 9); sun.castShadow = true; sun.shadow.mapSize.set(1024, 1024);
      var sc = sun.shadow.camera; sc.left = -11; sc.right = 11; sc.top = 9; sc.bottom = -9; sc.near = 1; sc.far = 40; sun.shadow.bias = -0.0006; scene.add(sun);
      var wl = new T.PointLight('#8fd8ff', 0.5, 14); wl.position.set(-5, 4, 1); scene.add(wl); var wr = new T.PointLight('#ffc9a0', 0.5, 14); wr.position.set(5, 4, -1); scene.add(wr);
      buildRoom();
      Object.keys(DEFAULT_AGENTS).forEach(function (id) {
        var d = DEFAULT_AGENTS[id], c = buildRobot(id, d.color, d.kind); c.position.set(d.pos[0], 0, d.pos[1]);
        c.rotation.y = Math.atan2(FACE_POINT.x - d.pos[0], FACE_POINT.z - d.pos[1]); c.userData.displayName = d.name; c.userData.color = d.color; c.userData.id = id;
        c.scale.setScalar(1.25); clinic.add(c); chars[id] = c;
      });
      Object.keys(DEFAULT_AGENTS).forEach(function (id) { buildDesk(id, DEFAULT_AGENTS[id]); });
      // màn hình theo dõi: treo tường trái + đứng cạnh bệnh nhân
      var wm = makeMonitor(3.4, 1.7, false); wm.position.set(-6.78, 3.3, 0.2); wm.rotation.y = Math.PI / 2; clinic.add(wm);
      var sm = makeMonitor(1.7, 1.05, true); sm.position.set(-1.9, 2.55, 0.9); sm.rotation.y = 0.25; clinic.add(sm);
      patient = buildPatient(); clinic.add(patient); patient.userData.displayName = 'Bạn'; patient.userData.color = '#2589e0';
      Clinic.patientReset();
      resize(); var h = homeView(); camera.position.copy(h.pos); controls.target.copy(h.tgt); applyControlLimits(); controls.update();
      if (global.ResizeObserver) new ResizeObserver(resize).observe(stage); window.addEventListener('resize', resize);
      canvas.addEventListener('pointerdown', function (e) { pointerDown = { x: e.clientX, y: e.clientY }; });
      canvas.addEventListener('pointerup', function (e) {
        if (!pointerDown || mode !== 'kg') return; var d = Math.hypot(e.clientX - pointerDown.x, e.clientY - pointerDown.y); pointerDown = null;
        if (d < 5) { var id = pickNode(e); if (id) { Clinic.kgSelect(id); if (onPick) onPick(id); } }
      });
      canvas.addEventListener('pointermove', function (e) {
        if (mode !== 'kg' || e.buttons) return; var id = pickNode(e); if (id !== hoverId) { hoverId = id; canvas.style.cursor = id ? 'pointer' : 'grab'; }
      });
      started = true; requestAnimationFrame(frame);
    },
    setMode: function (m) {
      if (m === mode) return; var wasKg = mode === 'kg'; mode = m; clinic.visible = (m !== 'kg'); if (kgGroup) kgGroup.visible = (m === 'kg');
      scene.background.set(m === 'kg' ? '#eaf3ff' : '#cfe8ff'); hideAllBubbles(); applyControlLimits();
      if (m === 'kg' || wasKg) { var h = homeView(); camera.position.copy(h.pos); controls.target.copy(h.tgt); tween = null; controls.update(); }
      if (m === 'life') flyLife(1000); else if (m === 'clinic' && !wasKg) flyHome(1000);
    },
    setAgents: function (list) {
      (list || []).forEach(function (a) {
        var c = chars[a.id]; if (!c) return; c.userData.displayName = a.name; c.userData.color = a.color || c.userData.color;
        if (c.userData.sign) c.userData.sign.userData.redraw(a.name, a.role);
      });
    },
    flyToChar: flyToChar, flyToBoard: flyToBoard, flyHome: flyHome, flyLife: flyLife,
    say: say, hideBubble: hideBubble, hideAllBubbles: hideAllBubbles, beam: beam,
    speak: function (id, on) {
      Object.keys(chars).forEach(function (k) { chars[k].userData.speaking = false; chars[k].userData.look = null; });
      if (patient) { patient.userData.speaking = false; }
      speakerId = on ? id : null;
      if (on) {
        var c = id === 'patient' ? patient : chars[id]; if (c) c.userData.speaking = true;
        Object.keys(chars).forEach(function (k) { if (k !== id && c) chars[k].userData.look = c; });
      }
    },
    work: function (id, on) { if (chars[id]) chars[id].userData.working = !!on; },
    stopAll: function () { Object.keys(chars).forEach(function (k) { chars[k].userData.speaking = false; chars[k].userData.working = false; chars[k].userData.look = null; }); alertState.until = 0; },
    alert: function (level, ms) {
      alertState.level = level; alertState.color = level === 'danger' ? '#ff2a2a' : level === 'info' ? '#2f9bff' : '#ffaa00'; alertState.until = performance.now() + (ms || 2500);
    },
    getCharPos: function (id) { var c = id === 'patient' ? patient : chars[id]; return c ? c.position.clone() : null; },
    /* bảng tường */
    setKG: function (data) {
      kgData = data; kgLayout2d = null;
      var lay = forceLayout(data.nodes, data.edges, 2, 5), minx = 1e9, maxx = -1e9, miny = 1e9, maxy = -1e9;
      lay.P.forEach(function (p) { minx = Math.min(minx, p[0]); maxx = Math.max(maxx, p[0]); miny = Math.min(miny, p[1]); maxy = Math.max(maxy, p[1]); });
      kgLayout2d = {}; data.nodes.forEach(function (n, i) { var p = lay.P[i]; kgLayout2d[n.id] = [(p[0] - minx) / (maxx - minx || 1), (p[1] - miny) / (maxy - miny || 1)]; });
      drawBoard(); buildKG(data);
    },
    highlightKg: function (ids) {
      kgHighlight = null; if (ids && ids.length) { kgHighlight = {}; ids.forEach(function (i) { kgHighlight[i] = true; }); }
      drawBoard(); applyKgHighlight();
    },
    kgSelect: function (id) { kgSelected = id; applyKgHighlight(); if (id && controls) controls.autoRotate = false; },
    kgFocus: function (id) {
      var m = kgNodeMeshes.filter(function (x) { return x.userData.id === id; })[0]; if (!m) return;
      var dir = m.position.clone().normalize(); if (dir.length() < 0.1) dir.set(0, 0, 1);
      flyTo(m.position.clone().addScaledVector(dir, 7).add(new T.Vector3(0, 1.2, 0)), m.position.clone(), 1100);
    },
    kgEdgeLabels: function (on) { kgEdgeLabels.forEach(function (l) { l.visible = on; }); },
    kgAutoRotate: function (on) { if (controls) controls.autoRotate = !!on; },
    onNodePick: function (cb) { onPick = cb; },
    /* màn hình đường huyết */
    setGlucose: function (g) { glucoseState = Object.assign({}, glucoseState, g); monitors.forEach(drawMonitor); },
    /* bệnh nhân */
    setPatient: function (o) {
      var u = patient.userData; if (o.name != null) { u.displayName = o.name || 'Bạn'; }
      if (o.color) { u.shirtM.color.set(o.color); u.color = o.color; }
      if (o.sex) { var f = o.sex === 'female'; u.longHair.visible = f; u.bun.visible = f; u.skirt.visible = f; u.hairM.color.set(f ? '#5a3320' : '#2e2420'); }
      if (o.bmi != null) {
        var k = clamp((o.bmi - 18) / 17, 0, 1); var sx = lerp(0.82, 1.38, k); u.torso.scale.set(sx, 1, lerp(0.85, 1.3, k)); u.belly.scale.set(0.9 + k * 0.25, 0.7 + k * 0.2, 0.6 + k * 0.5);
        u.skirt.scale.set(lerp(0.88, 1.3, k), 1, lerp(0.88, 1.3, k)); u.legs.forEach(function (l) { l.scale.x = lerp(0.9, 1.25, k); l.scale.z = lerp(0.9, 1.25, k); });
        u.arms.forEach(function (a, i) { a.position.x = (i ? 1 : -1) * 0.58 * sx; });
      }
    },
    patientReset: function () { patient.position.copy(PATIENT_DOOR); patient.rotation.y = -2.2; patient.userData.move = null; setPatientState('wave'); patient.visible = true; },
    patientEnter: function () {
      return new Promise(function (res) {
        patient.userData.move = { to: PATIENT_SPOT.clone(), speed: 2.4, faceYaw: -0.55, resolve: function () { setPatientState('wave'); setTimeout(function () { setPatientState('idle'); }, 1100); res(); } };
        setPatientState('walk');
      });
    },
    patientAtSpot: function () { patient.position.copy(PATIENT_SPOT); patient.rotation.y = -0.55; patient.userData.move = null; setPatientState('idle'); },
    patientAct: function (actionId, ms) {
      return new Promise(function (res) {
        var u = patient.userData, st = ACTION_STATE[actionId] || 'skip';
        if (patient.position.distanceTo(PATIENT_SPOT) > 0.3 && !u.move) { patient.position.copy(PATIENT_SPOT); patient.rotation.y = -0.55; }
        if (st === 'eat') setFood(u, actionId);
        setPatientState(st); patient.rotation.y = -0.2;
        setTimeout(function () { setPatientState('idle'); patient.rotation.y = -0.55; res(); }, ms || 1600);
      });
    },
    patientIdle: function () { setPatientState('idle'); },
    patientSpeaking: function (on) { patient.userData.speaking = !!on; },
    debug: function () { return { pos: camera.position.toArray(), tgt: controls.target.toArray(), W: W, H: H, aspect: camera.aspect, fov: camera.fov }; },
    resize: resize, isStarted: function () { return started; }, getMode: function () { return mode; }
  };
  global.Clinic = Clinic;
})(window);
