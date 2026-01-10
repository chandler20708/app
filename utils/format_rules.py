def format_rule_for_management(rule):
    conditions = []
    def format_value(raw):
        try:
            val = float(raw)
        except (TypeError, ValueError):
            return raw
        return int(val) if val.is_integer() else round(val, 1)

    def format_numeric(feature, op, raw):
        val = format_value(raw)
        if feature == "noofinvestigation":
            return f"**{val} or fewer investigations**" if op == "≤" else f"**more than {val} investigations**"
        if feature == "nooftreatment":
            return f"**{val} or fewer treatments**" if op == "≤" else f"**more than {val} treatments**"
        if feature == "noofpatients":
            return f"**{val} or fewer patients**" if op == "≤" else f"**more than {val} patients**"
        if feature == "age":
            return f"**age {op} {val}**"
        if feature == "los":
            return (
                f"**{val} hours or less in AED**"
                if op == "≤"
                else f"**more than {val} hours in AED**"
            )
        return f"**{feature} {op} {val}**"

    for c in rule["conditions"]:
        if " ≤ " in c:
            feature, raw = c.split(" ≤ ", 1)
            if feature.startswith("HRG_grp_"):
                feat = feature.replace("HRG_grp_", "")
                conditions.append(f"**Treatment category ≠ {feat}**")
            else:
                conditions.append(format_numeric(feature, "≤", raw))
        elif " > " in c:
            feature, raw = c.split(" > ", 1)
            if feature.startswith("HRG_grp_"):
                feat = feature.replace("HRG_grp_", "")
                conditions.append(f"**Treatment category = {feat}**")
            else:
                conditions.append(format_numeric(feature, ">", raw))

    risk = rule["breach_rate"] * 100
    icon = "🔴" if risk >= 80 else "🟠" if risk >= 65 else "🟡"

    if not conditions:
        conditions.append("**no dominant conditions identified**")

    return (
        f"{icon} **{risk:.0f}% breach risk** when "
        f"{', and '.join(conditions)} "
        f"*(n = {rule['samples']})*"
    )
