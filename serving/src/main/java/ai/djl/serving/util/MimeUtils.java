/*
 * Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance
 * with the License. A copy of the License is located at
 *
 * http://aws.amazon.com/apache2.0/
 *
 * or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
 * OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions
 * and limitations under the License.
 */
package ai.djl.serving.util;

import io.netty.handler.codec.http.HttpHeaderValues;
import io.netty.util.AsciiString;

import java.util.Locale;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/** A utility class that handling MIME types. */
public final class MimeUtils {

    private static final Map<String, AsciiString> MIME_TYPE_MAP = new ConcurrentHashMap<>();

    static {
        MIME_TYPE_MAP.put("htm", HttpHeaderValues.TEXT_HTML);
        MIME_TYPE_MAP.put("html", HttpHeaderValues.TEXT_HTML);
        MIME_TYPE_MAP.put("js", AsciiString.of("application/javascript"));
        MIME_TYPE_MAP.put("xml", HttpHeaderValues.APPLICATION_XML);
        MIME_TYPE_MAP.put("css", HttpHeaderValues.TEXT_CSS);
        MIME_TYPE_MAP.put("txt", HttpHeaderValues.TEXT_PLAIN);
        MIME_TYPE_MAP.put("text", HttpHeaderValues.TEXT_PLAIN);
        MIME_TYPE_MAP.put("log", HttpHeaderValues.TEXT_PLAIN);
        MIME_TYPE_MAP.put("csv", AsciiString.of("text/comma-separated-values"));
        MIME_TYPE_MAP.put("rtf", AsciiString.of("text/rtf"));
        MIME_TYPE_MAP.put("sh", AsciiString.of("text/x-sh"));
        MIME_TYPE_MAP.put("tex", AsciiString.of("application/x-tex"));
        MIME_TYPE_MAP.put("texi", AsciiString.of("application/x-texinfo"));
        MIME_TYPE_MAP.put("texinfo", AsciiString.of("application/x-texinfo"));
        MIME_TYPE_MAP.put("t", AsciiString.of("application/x-troff"));
        MIME_TYPE_MAP.put("tr", AsciiString.of("application/x-troff"));
        MIME_TYPE_MAP.put("roff", AsciiString.of("application/x-troff"));
        MIME_TYPE_MAP.put("gif", AsciiString.of("image/gif"));
        MIME_TYPE_MAP.put("png", AsciiString.of("image/x-png"));
        MIME_TYPE_MAP.put("ief", AsciiString.of("image/ief"));
        MIME_TYPE_MAP.put("jpeg", AsciiString.of("image/jpeg"));
        MIME_TYPE_MAP.put("jpg", AsciiString.of("image/jpeg"));
        MIME_TYPE_MAP.put("jpe", AsciiString.of("image/jpeg"));
        MIME_TYPE_MAP.put("tiff", AsciiString.of("image/tiff"));
        MIME_TYPE_MAP.put("tif", AsciiString.of("image/tiff"));
        MIME_TYPE_MAP.put("xwd", AsciiString.of("image/x-xwindowdump"));
        MIME_TYPE_MAP.put("pict", AsciiString.of("image/x-pict"));
        MIME_TYPE_MAP.put("bmp", AsciiString.of("image/x-ms-bmp"));
        MIME_TYPE_MAP.put("pcd", AsciiString.of("image/x-photo-cd"));
        MIME_TYPE_MAP.put("dwg", AsciiString.of("image/vnd.dwg"));
        MIME_TYPE_MAP.put("dxf", AsciiString.of("image/vnd.dxf"));
        MIME_TYPE_MAP.put("svf", AsciiString.of("image/vnd.svf"));
        MIME_TYPE_MAP.put("au", AsciiString.of("autio/basic"));
        MIME_TYPE_MAP.put("snd", AsciiString.of("autio/basic"));
        MIME_TYPE_MAP.put("mid", AsciiString.of("autio/midi"));
        MIME_TYPE_MAP.put("midi", AsciiString.of("autio/midi"));
        MIME_TYPE_MAP.put("aif", AsciiString.of("autio/x-aiff"));
        MIME_TYPE_MAP.put("aiff", AsciiString.of("autio/x-aiff"));
        MIME_TYPE_MAP.put("aifc", AsciiString.of("autio/x-aiff"));
        MIME_TYPE_MAP.put("wav", AsciiString.of("autio/x-wav"));
        MIME_TYPE_MAP.put("mpa", AsciiString.of("autio/x-mpeg"));
        MIME_TYPE_MAP.put("abs", AsciiString.of("autio/x-mpeg"));
        MIME_TYPE_MAP.put("mpega", AsciiString.of("autio/x-mpeg"));
        MIME_TYPE_MAP.put("mp2a", AsciiString.of("autio/x-mpeg-2"));
        MIME_TYPE_MAP.put("mpa2", AsciiString.of("autio/x-mpeg-2"));
        MIME_TYPE_MAP.put("ra", AsciiString.of("application/x-pn-realaudio"));
        MIME_TYPE_MAP.put("ram", AsciiString.of("application/x-pn-realaudio"));
        MIME_TYPE_MAP.put("mpeg", AsciiString.of("video/mpeg"));
        MIME_TYPE_MAP.put("mpg", AsciiString.of("video/mpeg"));
        MIME_TYPE_MAP.put("mpe", AsciiString.of("video/mpeg"));
        MIME_TYPE_MAP.put("mpv2", AsciiString.of("video/mpeg-2"));
        MIME_TYPE_MAP.put("mp2v", AsciiString.of("video/mpeg-2"));
        MIME_TYPE_MAP.put("qt", AsciiString.of("video/quicktime"));
        MIME_TYPE_MAP.put("mov", AsciiString.of("video/quicktime"));
        MIME_TYPE_MAP.put("avi", AsciiString.of("video/x-msvideo"));
        MIME_TYPE_MAP.put("ai", AsciiString.of("application/postscript"));
        MIME_TYPE_MAP.put("eps", AsciiString.of("application/postscript"));
        MIME_TYPE_MAP.put("ps", AsciiString.of("application/postscript"));
        MIME_TYPE_MAP.put("pdf", AsciiString.of("application/pdf"));
        MIME_TYPE_MAP.put("gtar", AsciiString.of("application/x-gtar"));
        MIME_TYPE_MAP.put("tar", AsciiString.of("application/x-tar"));
        MIME_TYPE_MAP.put("bcpio", AsciiString.of("application/x-bcpio"));
        MIME_TYPE_MAP.put("cpio", AsciiString.of("application/x-cpio"));
        MIME_TYPE_MAP.put("zip", AsciiString.of("application/zip"));
        MIME_TYPE_MAP.put("rar", AsciiString.of("application/rar"));
    }

    private MimeUtils() {}

    /**
     * Return the content type that associated with the file.
     *
     * @param fileType file extension
     * @return the content type
     */
    public static AsciiString getContentType(String fileType) {
        AsciiString contentType = MIME_TYPE_MAP.get(fileType.toLowerCase(Locale.ROOT));
        if (contentType == null) {
            return HttpHeaderValues.APPLICATION_OCTET_STREAM;
        }
        return contentType;
    }
}
