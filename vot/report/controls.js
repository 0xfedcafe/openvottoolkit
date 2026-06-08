
// Drives the sticky filter panel: toggle trackers (hides matching lines in every plot),
// toggle whole analysis subtypes, collapse individual items, reveal static heatmap twins.

$(function () {

    function applyTrackers() {
        $(".cp-trackers li").each(function () {
            var n = $(this).data("tracker");
            var off = $(this).hasClass("off");
            // Matplotlib writes the per-tracker gid as the SVG element id; the same id
            // repeats once per plot, so select every occurrence by attribute.
            $('[id="vottracker_' + n + '"]').css("display", off ? "none" : "");
        });
    }

    function applyKinds() {
        $(".cp-kinds li").each(function () {
            var kind = $(this).data("kind");
            var off = $(this).hasClass("off");
            $(".report-item").filter(function () {
                return $(this).attr("data-kind") === String(kind);
            }).toggleClass("kind-off", off);
        });
    }

    $(".cp-trackers li").click(function () {
        $(this).toggleClass("off");
        applyTrackers();
    });

    $(".cp-kinds li").click(function () {
        $(this).toggleClass("off");
        applyKinds();
    });

    $(".cp-actions a").click(function () {
        var off = $(this).data("act") === "none";
        var group = $(this).data("group");
        var list = group === "tracker" ? ".cp-trackers li" : ".cp-kinds li";
        $(list).toggleClass("off", off);
        if (group === "tracker") { applyTrackers(); } else { applyKinds(); }
    });

    $("#cp-collapse").click(function () {
        $("#control-panel").toggleClass("collapsed");
    });

    $(".item-collapse").click(function () {
        $(this).closest(".report-item").toggleClass("collapsed");
    });

    $("#cp-mpl-dup").change(function () {
        $("body").toggleClass("show-mpl-duplicates", this.checked);
    });

});
