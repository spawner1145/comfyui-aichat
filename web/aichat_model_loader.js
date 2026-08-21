import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

if (!window.__aichatModelLoaderRegistered) {
    window.__aichatModelLoaderRegistered = true;

    const PROVIDERS = {
        OpenAIApiLoader: {
            key: "openai",
            label: "OpenAI",
            route: "/aichat/api/openai/get_models",
        },
        GeminiApiLoader: {
            key: "gemini",
            label: "Gemini",
            route: "/aichat/api/gemini/get_models",
        },
    };

    let currentPanel = null;

    function injectStyles() {
        if (document.getElementById("aichat-model-loader-styles")) return;
        const style = document.createElement("style");
        style.id = "aichat-model-loader-styles";
        style.textContent = `
.aichat-panel { position: fixed; z-index: 10000; width: 540px; max-width: 94vw; max-height: 84vh; overflow: auto; background: var(--comfy-menu-bg, #202020); color: var(--fg-color, #e0e0e0); border: 1px solid var(--border-color, #444); border-radius: 10px; box-shadow: 0 12px 40px rgba(0,0,0,0.55); font-family: -apple-system, "Segoe UI", "Microsoft YaHei", sans-serif; font-size: 13px; }
.aichat-panel__header { position: sticky; top: 0; z-index: 2; display:flex; align-items:center; justify-content:space-between; min-height: 39px; box-sizing: border-box; padding:10px 14px; background: var(--comfy-menu-bg, #202020); border-bottom:1px solid var(--border-color, #444); }
.aichat-panel__body { padding: 14px; }
.aichat-panel__title { font-weight: 600; font-size: 14px; }
.aichat-close { background: transparent; color: inherit; border: none; font-size: 18px; cursor: pointer; }
.aichat-field { margin-bottom: 10px; }
.aichat-field label { display: block; margin-bottom: 4px; opacity: 0.8; font-size: 12px; }
.aichat-input-row { display: flex; gap: 8px; align-items: center; }
.aichat-input, .aichat-select { width: 100%; box-sizing: border-box; background: var(--comfy-input-bg, #2b2b2b); color: var(--input-text, #e0e0e0); border: 1px solid var(--border-color, #444); border-radius: 6px; padding: 7px 9px; font-size: 13px; }
.aichat-row { display: flex; gap: 8px; }
.aichat-row > * { flex: 1 1 0; }
.aichat-btn { cursor: pointer; border: 1px solid var(--border-color, #555); background: var(--comfy-input-bg, #333); color: var(--fg-color, #e0e0e0); border-radius: 6px; padding: 7px 12px; font-size: 13px; }
.aichat-btn:hover { background: #3a3a3a; }
.aichat-btn--primary { background: #2f6fdb; border-color: #2f6fdb; color: #fff; font-weight: 600; }
.aichat-btn--danger { background: #6b2d2d; border-color: #6b2d2d; color: #fff; }
.aichat-status { min-height: 18px; margin: 8px 0 12px; font-size: 12px; }
.aichat-status--ok { color: #6fcf7f; }
.aichat-status--err { color: #ff7a7a; }
.aichat-status--info { opacity: 0.75; }
.aichat-section { margin-top: 16px; padding-top: 12px; border-top: 1px solid rgba(255,255,255,0.08); }
.aichat-collapse { margin-top: 12px; }
.aichat-collapse__header { display: flex; align-items: center; gap: 8px; padding: 8px 10px; border: 1px solid var(--border-color, #444); border-radius: 6px; background: rgba(255,255,255,0.04); cursor: pointer; user-select: none; }
.aichat-collapse__header:hover { background: rgba(255,255,255,0.08); }
.aichat-collapse__arrow { width: 12px; opacity: 0.8; }
.aichat-collapse__title { font-weight: 600; }
.aichat-collapse--editor .aichat-collapse__header { position: sticky; top: 39px; z-index: 1; flex-wrap: wrap; background: var(--comfy-menu-bg, #202020); }
.aichat-collapse__actions { display: flex; gap: 8px; margin-left: auto; }
.aichat-collapse__actions .aichat-btn { padding: 5px 9px; }
.aichat-quick-add { margin-top: 8px; border-color: #2f6fdb; }
.aichat-quick-add__title { font-weight: 600; margin-bottom: 8px; }
.aichat-quick-add__actions { display: flex; gap: 8px; }
.aichat-quick-add__actions > * { flex: 1 1 0; }
.aichat-collapse__body { margin-top: 8px; }
.aichat-group-list { display: flex; flex-direction: column; gap: 8px; margin-top: 10px; }
.aichat-group-item { padding: 10px; border: 1px solid var(--border-color, #444); border-radius: 8px; background: rgba(255,255,255,0.03); }
.aichat-group-item__name { font-weight: 600; margin-bottom: 6px; }
.aichat-model-list { margin-top: 8px; border: 1px solid var(--border-color, #444); border-radius: 6px; max-height: 220px; overflow: auto; }
.aichat-model-item { padding: 7px 10px; cursor: pointer; border-bottom: 1px solid rgba(255,255,255,0.05); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.aichat-model-item:last-child { border-bottom: none; }
.aichat-model-item:hover { background: rgba(74,158,255,0.18); }
.aichat-empty { padding: 12px; opacity: 0.7; text-align: center; }
        `;
        document.head.appendChild(style);
    }

    function closePanel() {
        if (currentPanel) {
            currentPanel.remove();
            currentPanel = null;
        }
    }

    function getWidget(node, name) {
        return node.widgets?.find((w) => w.name === name) || null;
    }

    function getWidgetValue(node, name, fallback = "") {
        const w = getWidget(node, name);
        return w ? w.value : fallback;
    }

    function setWidgetValue(node, name, value) {
        const w = getWidget(node, name);
        if (!w) return;
        w.value = value;
        if (typeof w.callback === "function") {
            try {
                w.callback(value, app.canvas, node);
            } catch (e) {}
        }
        node.setDirtyCanvas(true, true);
    }

    async function postJson(route, body) {
        const resp = await api.fetchApi(route, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body),
        });
        return await resp.json();
    }

    async function loadGroups(providerKey) {
        return await postJson("/aichat/api/groups/get", { provider: providerKey });
    }

    async function saveGroups(providerKey, groups) {
        return await postJson("/aichat/api/groups/save", { provider: providerKey, groups });
    }

    async function fetchModels(node, provider) {
        return await postJson(provider.route, {
            api_key: getWidgetValue(node, "api_key"),
            base_url: getWidgetValue(node, "base_url"),
            channel_group: getWidgetValue(node, "channel_group"),
            proxy_http: getWidgetValue(node, "proxy_http"),
            proxy_https: getWidgetValue(node, "proxy_https"),
            timeout: getWidgetValue(node, "timeout", 60),
        });
    }

    function makeField(labelText, type = "text") {
        const field = document.createElement("div");
        field.className = "aichat-field";
        const label = document.createElement("label");
        label.textContent = labelText;
        const row = document.createElement("div");
        row.className = "aichat-input-row";
        const input = type === "select" ? document.createElement("select") : document.createElement("input");
        input.className = type === "select" ? "aichat-select" : "aichat-input";
        if (type !== "select") input.type = type;
        field.appendChild(label);
        row.appendChild(input);
        field.appendChild(row);
        return { field, input, row };
    }

    function addPasswordToggle(fieldObj) {
        if (!fieldObj?.input || fieldObj.input.tagName !== "INPUT" || fieldObj.input.type !== "password") {
            return;
        }
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = "aichat-btn";
        btn.textContent = "显示";
        btn.onclick = () => {
            const isPassword = fieldObj.input.type === "password";
            fieldObj.input.type = isPassword ? "text" : "password";
            btn.textContent = isPassword ? "隐藏" : "显示";
        };
        fieldObj.row.appendChild(btn);
    }

    function renderModels(container, models, onPick, filterText = "") {
        container.innerHTML = "";
        const list = Array.isArray(models) ? models : [];
        const keyword = (filterText || "").trim().toLowerCase();
        const filtered = keyword ? list.filter((model) => model.toLowerCase().includes(keyword)) : list;

        if (!list.length) {
            const empty = document.createElement("div");
            empty.className = "aichat-empty";
            empty.textContent = "暂无模型，请先获取模型列表";
            container.appendChild(empty);
            return;
        }

        if (!filtered.length) {
            const empty = document.createElement("div");
            empty.className = "aichat-empty";
            empty.textContent = "没有匹配的模型";
            container.appendChild(empty);
            return;
        }

        for (const model of filtered) {
            const item = document.createElement("div");
            item.className = "aichat-model-item";
            item.textContent = model;
            item.title = model;
            item.onclick = () => onPick(model);
            container.appendChild(item);
        }
    }

    async function openConfigPanel(node, provider) {
        injectStyles();
        closePanel();

        const panel = document.createElement("div");
        panel.className = "aichat-panel";
        currentPanel = panel;

        const header = document.createElement("div");
        header.className = "aichat-panel__header";
        const title = document.createElement("div");
        title.className = "aichat-panel__title";
        title.textContent = `${provider.label} 渠道组与模型配置`;
        const closeBtn = document.createElement("button");
        closeBtn.className = "aichat-close";
        closeBtn.textContent = "x";
        closeBtn.onclick = closePanel;
        header.appendChild(title);
        header.appendChild(closeBtn);
        panel.appendChild(header);

        const body = document.createElement("div");
        body.className = "aichat-panel__body";
        panel.appendChild(body);

        const status = document.createElement("div");
        status.className = "aichat-status aichat-status--info";
        status.textContent = "支持直接填写 base_url/api_key，也支持选择渠道组。选择渠道组后，运行时优先使用渠道组。";

        const groupField = makeField("渠道组", "select");
        const groupPickerFilterField = makeField("筛选渠道组");
        groupPickerFilterField.input.placeholder = "输入渠道组名称筛选";
        const directBaseField = makeField("直接 Base URL");
        const directKeyField = makeField("直接 API Key", "password");
        const modelField = makeField("模型");
        const proxyHttpField = makeField("Proxy HTTP");
        const proxyHttpsField = makeField("Proxy HTTPS");
        addPasswordToggle(directKeyField);

        directBaseField.input.value = getWidgetValue(node, "base_url");
        directKeyField.input.value = getWidgetValue(node, "api_key");
        modelField.input.value = getWidgetValue(node, "model");
        proxyHttpField.input.value = getWidgetValue(node, "proxy_http");
        proxyHttpsField.input.value = getWidgetValue(node, "proxy_https");

        directBaseField.input.oninput = () => setWidgetValue(node, "base_url", directBaseField.input.value);
        directKeyField.input.oninput = () => setWidgetValue(node, "api_key", directKeyField.input.value);
        modelField.input.oninput = () => setWidgetValue(node, "model", modelField.input.value);
        proxyHttpField.input.oninput = () => setWidgetValue(node, "proxy_http", proxyHttpField.input.value);
        proxyHttpsField.input.oninput = () => setWidgetValue(node, "proxy_https", proxyHttpsField.input.value);

        body.appendChild(groupField.field);
        body.appendChild(groupPickerFilterField.field);
        body.appendChild(directBaseField.field);
        body.appendChild(directKeyField.field);
        body.appendChild(modelField.field);
        body.appendChild(proxyHttpField.field);
        body.appendChild(proxyHttpsField.field);
        body.appendChild(status);

        const buttonRow = document.createElement("div");
        buttonRow.className = "aichat-row";
        const getModelsBtn = document.createElement("button");
        getModelsBtn.className = "aichat-btn aichat-btn--primary";
        getModelsBtn.textContent = "获取模型列表";
        const reloadGroupsBtn = document.createElement("button");
        reloadGroupsBtn.className = "aichat-btn";
        reloadGroupsBtn.textContent = "刷新渠道组";
        buttonRow.appendChild(getModelsBtn);
        buttonRow.appendChild(reloadGroupsBtn);
        body.appendChild(buttonRow);

        const modelCollapse = document.createElement("div");
        modelCollapse.className = "aichat-collapse";
        const modelCollapseHeader = document.createElement("div");
        modelCollapseHeader.className = "aichat-collapse__header";
        const modelCollapseArrow = document.createElement("span");
        modelCollapseArrow.className = "aichat-collapse__arrow";
        const modelCollapseTitle = document.createElement("span");
        modelCollapseTitle.className = "aichat-collapse__title";
        modelCollapseHeader.appendChild(modelCollapseArrow);
        modelCollapseHeader.appendChild(modelCollapseTitle);
        modelCollapse.appendChild(modelCollapseHeader);

        const modelCollapseBody = document.createElement("div");
        modelCollapseBody.className = "aichat-collapse__body";
        modelCollapse.appendChild(modelCollapseBody);

        const modelFilterField = makeField("筛选模型");
        modelFilterField.input.placeholder = "输入关键字快速筛选模型";
        modelCollapseBody.appendChild(modelFilterField.field);

        const modelList = document.createElement("div");
        modelList.className = "aichat-model-list";
        modelCollapseBody.appendChild(modelList);
        body.appendChild(modelCollapse);

        const groupSection = document.createElement("div");
        groupSection.className = "aichat-section";
        body.appendChild(groupSection);

        const groupCollapse = document.createElement("div");
        groupCollapse.className = "aichat-collapse aichat-collapse--editor";
        const groupCollapseHeader = document.createElement("div");
        groupCollapseHeader.className = "aichat-collapse__header";
        const groupCollapseArrow = document.createElement("span");
        groupCollapseArrow.className = "aichat-collapse__arrow";
        const groupCollapseTitle = document.createElement("span");
        groupCollapseTitle.className = "aichat-collapse__title";
        const groupActions = document.createElement("div");
        groupActions.className = "aichat-collapse__actions";
        groupActions.onclick = (event) => event.stopPropagation();

        const addGroupBtn = document.createElement("button");
        addGroupBtn.className = "aichat-btn";
        addGroupBtn.textContent = "新增渠道组";
        const saveGroupBtn = document.createElement("button");
        saveGroupBtn.className = "aichat-btn aichat-btn--primary";
        saveGroupBtn.textContent = "保存到 YAML";
        groupActions.appendChild(addGroupBtn);
        groupActions.appendChild(saveGroupBtn);
        groupCollapseHeader.appendChild(groupCollapseArrow);
        groupCollapseHeader.appendChild(groupCollapseTitle);
        groupCollapseHeader.appendChild(groupActions);
        groupCollapse.appendChild(groupCollapseHeader);

        const quickAddEditor = document.createElement("div");
        quickAddEditor.className = "aichat-group-item aichat-quick-add";
        quickAddEditor.style.display = "none";
        const quickAddTitle = document.createElement("div");
        quickAddTitle.className = "aichat-quick-add__title";
        quickAddTitle.textContent = "快速新增渠道组";
        quickAddEditor.appendChild(quickAddTitle);

        const quickNameField = makeField("名称");
        const quickBaseField = makeField("Base URL");
        const quickKeyField = makeField("API Key", "password");
        addPasswordToggle(quickKeyField);
        quickAddEditor.appendChild(quickNameField.field);
        quickAddEditor.appendChild(quickBaseField.field);
        quickAddEditor.appendChild(quickKeyField.field);

        const quickAddActions = document.createElement("div");
        quickAddActions.className = "aichat-quick-add__actions";
        const quickConfirmBtn = document.createElement("button");
        quickConfirmBtn.className = "aichat-btn aichat-btn--primary";
        quickConfirmBtn.textContent = "确认新增";
        const quickCancelBtn = document.createElement("button");
        quickCancelBtn.className = "aichat-btn";
        quickCancelBtn.textContent = "取消";
        quickAddActions.appendChild(quickConfirmBtn);
        quickAddActions.appendChild(quickCancelBtn);
        quickAddEditor.appendChild(quickAddActions);
        groupCollapse.appendChild(quickAddEditor);

        const groupCollapseBody = document.createElement("div");
        groupCollapseBody.className = "aichat-collapse__body";
        groupCollapse.appendChild(groupCollapseBody);

        const groupFilterField = makeField("筛选渠道组");
        groupFilterField.input.placeholder = "输入名称或 Base URL 筛选渠道组";
        groupCollapseBody.appendChild(groupFilterField.field);

        const groupList = document.createElement("div");
        groupList.className = "aichat-group-list";
        groupCollapseBody.appendChild(groupList);
        groupSection.appendChild(groupCollapse);

        let groups = [];

        function hideQuickAddEditor() {
            quickAddEditor.style.display = "none";
        }

        function showQuickAddEditor() {
            quickNameField.input.value = "";
            quickBaseField.input.value = "";
            quickKeyField.input.value = "";
            quickAddEditor.style.display = "";
            quickNameField.input.focus();
        }

        function updateModelCollapseTitle() {
            const count = Array.isArray(node._aichatModels) ? node._aichatModels.length : 0;
            modelCollapseTitle.textContent = `模型列表 (${count})`;
        }

        function updateGroupCollapseTitle() {
            groupCollapseTitle.textContent = `编辑渠道组 (${groups.length})`;
        }

        function applyModelCollapseState() {
            const collapsed = !!node._aichatModelListCollapsed;
            modelCollapseArrow.textContent = collapsed ? "▶" : "▼";
            modelCollapseBody.style.display = collapsed ? "none" : "";
        }

        function applyGroupCollapseState() {
            const collapsed = !!node._aichatGroupEditorCollapsed;
            groupCollapseArrow.textContent = collapsed ? "▶" : "▼";
            groupCollapseBody.style.display = collapsed ? "none" : "";
        }

        function refreshModelList() {
            updateModelCollapseTitle();
            renderModels(
                modelList,
                node._aichatModels || [],
                (name) => {
                    setWidgetValue(node, "model", name);
                    modelField.input.value = name;
                    status.className = "aichat-status aichat-status--ok";
                    status.textContent = `已选择模型: ${name}`;
                },
                modelFilterField.input.value
            );
        }

        function syncGroupSelect() {
            const selected = getWidgetValue(node, "channel_group");
            const keyword = (groupPickerFilterField.input.value || "").trim().toLowerCase();
            const selectedGroup = groups.find((group) => group.name === selected);
            const visibleGroups = keyword
                ? groups.filter(
                      (group) =>
                          group === selectedGroup || String(group.name || "").toLowerCase().includes(keyword)
                  )
                : groups;
            groupField.input.innerHTML = "";

            const empty = document.createElement("option");
            empty.value = "";
            empty.textContent = "不使用渠道组，直接走下方 base_url/api_key";
            groupField.input.appendChild(empty);

            for (const group of visibleGroups) {
                const option = document.createElement("option");
                option.value = group.name;
                option.textContent = group.name;
                groupField.input.appendChild(option);
            }
            groupField.input.value = visibleGroups.some((g) => g.name === selected) ? selected : "";
        }

        function renderGroupEditors() {
            groupList.innerHTML = "";
            updateGroupCollapseTitle();
            const keyword = (groupFilterField.input.value || "").trim().toLowerCase();
            const filteredGroups = keyword
                ? groups.filter((group) =>
                      [group.name, group.base_url].some((value) =>
                          String(value || "").toLowerCase().includes(keyword)
                      )
                  )
                : groups;

            if (!filteredGroups.length) {
                const empty = document.createElement("div");
                empty.className = "aichat-empty";
                empty.textContent = groups.length ? "没有匹配的渠道组" : "暂无渠道组，请先新增或刷新";
                groupList.appendChild(empty);
                return;
            }

            for (const group of filteredGroups) {
                const item = document.createElement("div");
                item.className = "aichat-group-item";

                const nameField = makeField("名称");
                const baseField = makeField("Base URL");
                const keyField = makeField("API Key", "password");
                addPasswordToggle(keyField);
                nameField.input.value = group.name || "";
                baseField.input.value = group.base_url || "";
                keyField.input.value = group.api_key || "";

                nameField.input.oninput = () => {
                    group.name = nameField.input.value.trim();
                    syncGroupSelect();
                    titleEl.textContent = group.name || "未命名渠道组";
                };
                baseField.input.oninput = () => {
                    group.base_url = baseField.input.value;
                };
                keyField.input.oninput = () => {
                    group.api_key = keyField.input.value;
                };

                const delBtn = document.createElement("button");
                delBtn.className = "aichat-btn aichat-btn--danger";
                delBtn.textContent = "删除";
                delBtn.onclick = () => {
                    groups = groups.filter((g) => g !== group);
                    if (getWidgetValue(node, "channel_group") === group.name) {
                        setWidgetValue(node, "channel_group", "");
                    }
                    syncGroupSelect();
                    renderGroupEditors();
                };

                const titleEl = document.createElement("div");
                titleEl.className = "aichat-group-item__name";
                titleEl.textContent = group.name || "未命名渠道组";

                item.appendChild(titleEl);
                item.appendChild(nameField.field);
                item.appendChild(baseField.field);
                item.appendChild(keyField.field);
                item.appendChild(delBtn);
                groupList.appendChild(item);
            }
        }

        async function refreshGroups() {
            status.className = "aichat-status aichat-status--info";
            status.textContent = "正在读取渠道组...";
            const data = await loadGroups(provider.key);
            if (data.error) {
                status.className = "aichat-status aichat-status--err";
                status.textContent = "读取渠道组失败: " + data.error;
                return;
            }
            groups = Array.isArray(data.groups) ? data.groups.map((x) => ({ ...x })) : [];
            syncGroupSelect();
            renderGroupEditors();
            status.className = "aichat-status aichat-status--ok";
            status.textContent = `已读取 ${groups.length} 个渠道组`;
        }

        groupField.input.onchange = () => {
            setWidgetValue(node, "channel_group", groupField.input.value);
        };

        modelCollapseHeader.onclick = () => {
            node._aichatModelListCollapsed = !node._aichatModelListCollapsed;
            applyModelCollapseState();
        };

        groupCollapseHeader.onclick = () => {
            node._aichatGroupEditorCollapsed = !node._aichatGroupEditorCollapsed;
            applyGroupCollapseState();
        };

        modelFilterField.input.oninput = refreshModelList;
        groupPickerFilterField.input.oninput = syncGroupSelect;
        groupFilterField.input.oninput = renderGroupEditors;

        addGroupBtn.onclick = () => {
            showQuickAddEditor();
        };

        quickCancelBtn.onclick = hideQuickAddEditor;

        quickConfirmBtn.onclick = () => {
            const group = {
                name: quickNameField.input.value.trim(),
                base_url: quickBaseField.input.value.trim(),
                api_key: quickKeyField.input.value.trim(),
            };
            if (!group.name || !group.base_url || !group.api_key) {
                status.className = "aichat-status aichat-status--err";
                status.textContent = "请填写名称、Base URL 和 API Key";
                return;
            }
            if (groups.some((item) => (item.name || "").trim() === group.name)) {
                status.className = "aichat-status aichat-status--err";
                status.textContent = `存在重复渠道组名称: ${group.name}`;
                return;
            }
            groups.push(group);
            hideQuickAddEditor();
            syncGroupSelect();
            renderGroupEditors();
            status.className = "aichat-status aichat-status--ok";
            status.textContent = "渠道组已新增，请点击保存到 YAML";
        };

        saveGroupBtn.onclick = async () => {
            if (quickAddEditor.style.display !== "none") {
                status.className = "aichat-status aichat-status--err";
                status.textContent = "请先确认或取消快速新增的渠道组";
                return;
            }
            const cleaned = groups
                .map((g) => ({
                    name: (g.name || "").trim(),
                    base_url: (g.base_url || "").trim(),
                    api_key: (g.api_key || "").trim(),
                }))
                .filter((g) => g.name && g.base_url && g.api_key);
            const names = new Set();
            for (const group of cleaned) {
                if (names.has(group.name)) {
                    status.className = "aichat-status aichat-status--err";
                    status.textContent = `存在重复渠道组名称: ${group.name}`;
                    return;
                }
                names.add(group.name);
            }
            const data = await saveGroups(provider.key, cleaned);
            if (data.error) {
                status.className = "aichat-status aichat-status--err";
                status.textContent = "保存失败: " + data.error;
                return;
            }
            groups = Array.isArray(data.groups) ? data.groups.map((x) => ({ ...x })) : [];
            const selected = getWidgetValue(node, "channel_group");
            if (selected && !groups.some((g) => g.name === selected)) {
                setWidgetValue(node, "channel_group", "");
            }
            syncGroupSelect();
            renderGroupEditors();
            status.className = "aichat-status aichat-status--ok";
            status.textContent = "渠道组已保存到 yaml";
        };

        reloadGroupsBtn.onclick = refreshGroups;

        getModelsBtn.onclick = async () => {
            status.className = "aichat-status aichat-status--info";
            status.textContent = "正在获取模型列表...";
            const data = await fetchModels(node, provider);
            if (data.error) {
                status.className = "aichat-status aichat-status--err";
                status.textContent = "获取模型失败: " + data.error;
                return;
            }
            node._aichatModels = data.models || [];
            node._aichatModelListCollapsed = false;
            applyModelCollapseState();
            refreshModelList();
            status.className = "aichat-status aichat-status--ok";
            status.textContent = `成功获取 ${node._aichatModels.length} 个模型`;
        };

        applyModelCollapseState();
        refreshModelList();
        applyGroupCollapseState();
        renderGroupEditors();

        document.body.appendChild(panel);
        const rect = panel.getBoundingClientRect();
        panel.style.left = Math.max(8, (window.innerWidth - rect.width) / 2) + "px";
        panel.style.top = Math.max(20, window.innerHeight * 0.08) + "px";

        await refreshGroups();
    }

    app.registerExtension({
        name: "aichat.model_loader",
        async beforeRegisterNodeDef(nodeType, nodeData) {
            const provider = PROVIDERS[nodeData.name];
            if (!provider) return;

            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
                this._aichatModels = this._aichatModels || [];

                const btn = this.addWidget("button", "配置渠道组 / 获取模型", "", () => openConfigPanel(this, provider));
                btn.serialize = false;
                this.setDirtyCanvas(true, true);
                return r;
            };
        },
    });
}
