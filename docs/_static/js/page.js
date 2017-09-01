/* Generate url tracking for each page */
var protocol = location.protocol.concat("//");
var host = protocol.concat(window.location.host);
var path = window.location.pathname;
var pathArr = path.split('/');
var icon = '<i class="fa fa-angle-right" aria-hidden="true"></i>';
var urlTracker = "<ul><li><a href=" + host + ">MXNet</a>" + icon + "</li>";

// Check whether this is another version
var lastUrl = host;
var versionIDX = -1;
for (var i = 1; i < pathArr.length; ++i) {
    lastUrl += '/' + pathArr[i];
    if(pathArr[i] == 'versions') {
        versionIDX = i;
        lastUrl += '/' + pathArr[i] + '/' + pathArr[i + 1];
        break;
    }
}
if (versionIDX > 0) {
    pathArr = pathArr.slice(versionIDX + 1, pathArr.length);
    urlTracker = "<ul><li><a href=" + lastUrl + "/index.html>MXNet</a>" + icon + "</li>";
}
else lastUrl = host;

for (var i = 1; i < pathArr.length; ++i) {
    if (pathArr[i] == 'index.html' || pathArr[i].length == 0) continue;
    if (pathArr[i].indexOf('#') != -1) pathArr[i] = pathArr[i].substring(0, pathArr[i].indexOf('#'));
    lastUrl += '/' + pathArr[i];
    if (pathArr[i].endsWith('.html')) pathArr[i] = pathArr[i].substring(0, pathArr[i].length - 5);
    if (i == pathArr.length - 1 || pathArr[i + 1].length == 0 ||
        pathArr[i + 1] == 'index.html') urlTracker += "<li><a class='last-tracker' href=" + lastUrl + ">" + pathArr[i] + "</a></li>";
    else {
        // Check whether current folder has index.html.
        // If it doesn't, disable the link.
        $.ajax(lastUrl + '/index.html', {
            type: "GET",
            statusCode: {
                404: function (response) {
                    urlTracker += "<li>" + pathArr[i] + icon + "</li>";
                }
            }, 
            success: function () {
                urlTracker += "<li><a href=" + lastUrl + '/index.html' + ">" + pathArr[i] + "</a>" + icon + "</li>";
            },
            async: false
        });
    }
}
urlTracker += '</ul>';
$('.page-tracker').append(urlTracker);

/* Generate top download btn*/
if ($('div.download-btn').length > 0) {
    var topBtn = $('div.download-btn').clone();
    topBtn.addClass('download-btn-top');
    topBtn.insertAfter(".page-tracker");
}