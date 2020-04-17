////////////////////////////////////////////////////////////////////
////////////////////////                  //////////////////////////
////////////////////////    Node graph    //////////////////////////
////////////////////////                  //////////////////////////
////////////////////////////////////////////////////////////////////
function drawd3graph(divid, graphdata, isDetailed, curoid) {

	var color = d3.scale.category20();

	var divarea = d3.select(divid);
	divarea.style("visibility", "hidden");
	setTimeout(function() {
		divarea.style("visibility", "visible");
	}, 1500)

	var graph_area_width = divarea.offsetWidth;
	var graph_area_height = divarea.offsetHeight;

	var divcx = 100;
	if(graph_area_width != null){
		divcx = graph_area_width / 2;
	}

	var divcy = 100;
	if(graph_area_height != null){
		divcy = graph_area_height / 2;
	}

	var force = d3.layout.force().charge(-2500).gravity(0.9).friction(0.5)
			.linkDistance(divcy);

	var svg = divarea.append("svg").attr("width", "100%")
			.attr("height", "100%");
	svg.attr("viewBox", "0 0 " + (divcx * 2 + 30) + " " + (divcy * 2 + 30));

	for (i = 0; i < graphdata["data"]["edges"].length; i++) {
		graphdata["data"]["edges"][i]["source"] = parseInt(graphdata["data"]["edges"][i]["source"]);
		graphdata["data"]["edges"][i]["target"] = parseInt(graphdata["data"]["edges"][i]["target"]);
	}

	graphdata["data"]["nodes"][0]["weight"] = 3;
	for (i = 1; i < graphdata["data"]["nodes"].length; i++) {

		graphdata["data"]["nodes"][i]["weight"] = 1;
	}

	var graph = {
		"nodes" : graphdata["data"]["nodes"],
		"links" : graphdata["data"]["edges"]
	};

	force.nodes(graph.nodes).links(graph.links).start();
	// force.nodes().attr("cx", 0).attr("cy",0);

	var link = svg.selectAll(".link").data(graph.links).enter().append("line")
			.attr("class", "link").style("stroke", "#555555").style(
					"stroke-width", 3);

	var nodeCircle = svg.selectAll(".node").data(graph.nodes).enter();
	var node = nodeCircle.append("circle").attr("class", "node").attr("r", 20)
			.style("stroke", "#FBE5D6").style("stroke-width", 3).style("fill",
					function(d) {
						switch (d.type) {
						case "query":
							return "#0B94B1";
						case "gene/protein":
							return "#AA4346";
						case "pathway":
							return "#dddd00";
						case "drug":
							return "#33da0a";
						case "disease/symptom":
							return "#dd00dd";
						case "transcription_factor":
							return "#aaaaaa";
						case "miRNA":
							return "#094412";
						case "gene_ontology_cc":
						case "gene_ontology_mf":
						case "gene_ontology_bp":
							return "#a533b9";
						}
					}).call(force.drag).on(
					"click",
					function(d) {
						if (isDetailed == 1) {
//							ajaxGetSnippet("ajaxSnippetArea", "ajaxQueryDiv", d.label, curoid);
							getSnippetAjax('snippets', d.label, curoid, 0, rows4ngrams);
						} else {

						}
					}).on("mouseover", function() {
				this.style.cursor = "pointer";
			});
	node.append("title").text(function(d) {
		return d.label;
	});

	var nodeLabel = nodeCircle.append("svg:text").attr("fill", "black").style(
			"text-anchor", "middle").style("font-weight", "bold").text(
			function(d) {
				if (d.label.length > 10)
					return d.label.substring(0, 10) + "...";
				else
					return d.label;
			}).call(force.drag).style("font-size", "13px")
			.on(
					"click",
					function(d) {
						if (isDetailed == 1) {
//							ajaxGetSnippet("ajaxSnippetArea", "ajaxQueryDiv", d.label, curoid);
							getSnippetAjax('snippets', d.label, curoid, 0, rows4ngrams);
						} else {
							// var mainqueryT =
							// document.getElementById("mainqtextbox");
							// var curquery = mainqueryT.value;
							// var querywordArr = curquery.split(" ");
							// if( querywordArr.indexOf(d.label) == -1) {
							// querywordArr.push(d.label);
							// }
							// var newQuery = querywordArr.join("+");

							var newQuery = d.label;

							location.href = "collection1/s?q=" + newQuery + "&t=l";
						}
					}).on("mouseover", function() {
				this.style.cursor = "pointer";
			});
	nodeLabel.append("title").text(function(d) {
		return d.label;
	});

	force.on("tick", function() {
		link.attr("x1", function(d) {
			var posx = d.source.x + divcx;
			return Math.max(Math.min(divcx * 2,posx),20);
		}).attr("y1", function(d) {
			var posy = d.source.y + divcy
			return Math.max(Math.min(divcy * 2,posy),20);
		}).attr("x2", function(d) {
			var posx = d.target.x + divcx;
			return Math.max(Math.min(divcx * 2,posx),20);
		}).attr("y2", function(d) {
			var posy = d.target.y + divcy;
			return Math.max(Math.min(divcy *2,posy),20);
		});

		node.attr("cx", function(d) {
			return Math.max(Math.min(divcx * 2, d.x + divcx), 20);
		}).attr("cy", function(d) {
			return Math.max(Math.min(divcy * 2, d.y + divcy), 20);
		});

		nodeLabel.attr("dx", function(d) {
			return Math.max(Math.min(divcx * 2, d.x + divcx), 20);
		}).attr("dy", function(d) {
			return Math.max(Math.min(divcy * 2, d.y + divcy), 20);
		});
	});

}

function drawCytoGraph(divid, graphdata, isDetailed, curoid) {

	function handle_click(event) {
		var target = event.target;
		// alert(target.data["label"] + " is clicked!");
//		ajaxGetSnippet("ajaxSnippetArea", "ajaxQueryDiv", target.data["label"], curoid);
		getSnippetAjax('snippets', target.data["label"], curoid, 0, rows4ngrams);
	}

	var div_id = divid;

	var network_json = graphdata;

	var nodeShape = {
		discreteMapper : {
			attrName : "type",
			entries : [ {
				attrValue : "query",
				value : "circle"
			}, {
				attrValue : "gene/protein",
				value : "OCTAGON"
			}, {
				attrValue : "pathway",
				value : "roundrect"
			}, {
				attrValue : "drug",
				value : "hexagon"
			}, {
				attrValue : "disease/symptom",
				value : "diamond"
			}, {
				attrValue : "transcription_factor",
				value : "OCTAGON"
			}, {
				attrValue : "miRNA",
				value : "OCTAGON"
			} ]
		}
	};

	var nodeColor = {
		discreteMapper : {
			attrName : "type",
			entries : [ {
				attrValue : "query",
				value : "#0B94B1"
			}, {
				attrValue : "gene/protein",
				value : "#9A0B0B"
			}, {
				attrValue : "pathway",
				value : "#dddd00"
			}, {
				attrValue : "drug",
				value : "#33da0a"
			}, {
				attrValue : "disease/symptom",
				value : "#dd00dd"
			}, {
				attrValue : "transcription_factor",
				value : "#aaaaaa"
			}, {
				attrValue : "miRNA",
				value : "#222200"
			} ]
		}
	};

	var visual_style = {
		global : {
			backgroundColor : "#ffffff"
		},
		nodes : {
			// shape : "circle",
			shape : nodeShape,
			borderWidth : 0,
			borderColor : "#ffffff",
			size : {
				defaultValue : 50
			},
			labelFontSize : 25,
			labelFontWeight : "bold",
			color : nodeColor,
			labelHorizontalAnchor : "center"

		},
		edges : {
			width : 3,
			color : "#0B94B1"
		}
	};

	// initialization options
	var options = {
		swfPath : "./swf/CytoscapeWeb",
		flashInstallerPath : "./swf/playerProductInstall"
	};

	var vis = new org.cytoscapeweb.Visualization(div_id, options);

	// callback when Cytoscape Web has finished drawing
	vis.ready(function() {

		// add a listener for when nodes and edges are clicked

		if (isDetailed == 1) {
			vis.addListener("click", "nodes", function(event) {
				handle_click(event);
			});
		}

	});

	// var layoutStyle = { name:"radius", options:{ radius:100 } };
	var layoutStyle = {
		name : "circle"
	};

	var draw_options = {

		network : network_json,

		panZoomControlVisible : false,

		layout : layoutStyle,

		visualStyle : visual_style,
		edgesMerged : true

	};

	vis.draw(draw_options);
}

// //////////////////////////////////////////////////////////////////
// ////////////////////// //////////////////////////
// ////////////////////// Term cloud //////////////////////////
// ////////////////////// //////////////////////////
// //////////////////////////////////////////////////////////////////

function convertGOterms(goterms, topn) {
	var words = [];
	var bp = goterms["bpterms"];
	var cc = goterms["ccterms"];
	var mf = goterms["mfterms"];

	if (bp && (topn == 0 || topn > bp.length))
		topn = bp.length;
	else if(!bp)
		topn = 0;

	var maxSize = 0;
	var minSize = 100000;

	for (i = 0; i < topn; i++) {
		singleword = {
			"word" : bp[i]["term"],
			"freq" : bp[i]["count"],
			"rank" : i,
			"type" : 3
		};
		words.push(singleword);

		if (maxSize < bp[i]["count"]) {
			maxSize = bp[i]["count"];
		} else if (minSize > bp[i]["count"]) {
			minSize = bp[i]["count"];
		}
	}

	if (cc && (topn == 0 || topn > cc.length))
		topn = cc.length;
	else if(!cc)
		topn = 0;

	for (i = 0; i < topn; i++) {
		singleword = {
			"word" : cc[i]["term"],
			"freq" : cc[i]["count"],
			"rank" : i,
			"type" : 2
		};
		words.push(singleword);

		if (maxSize < cc[i]["count"]) {
			maxSize = cc[i]["count"];
		} else if (minSize > cc[i]["count"]) {
			minSize = cc[i]["count"];
		}
	}

	if (mf && (topn == 0 || topn > mf.length))
		topn = mf.length;
	else if(!mf)
		topn = 0;

	for (i = 0; i < topn; i++) {
		singleword = {
			"word" : mf[i]["term"],
			"freq" : mf[i]["count"],
			"rank" : i,
			"type" : 1
		};
		words.push(singleword);

		if (maxSize < mf[i]["count"]) {
			maxSize = mf[i]["count"];
		} else if (minSize > mf[i]["count"]) {
			minSize = mf[i]["count"];
		}
	}

	return {
		"words" : words,
		"max" : maxSize,
		"min" : minSize
	};
}

function drawD3WordCloud(goterms, svgid, topn, isDetailed) {

	var converted = convertGOterms(goterms, topn);
	var words = converted["words"];
	var maxSize = converted["max"] * 3;
	var minSize = converted["min"];
	var gap = maxSize - minSize;
	var fill = d3.scale.category10();

	var freq_max = 10; // to normalize with max value
	var horizontalTopN = topn * 0.3; //

	var firstSVG = d3.select("#" + svgid);
	var svgWidth = firstSVG.attr("width");
	var svgHeight = firstSVG.attr("height");

	var oid = svgid.substring(10);

	function draw(words) {

		var childSVG = firstSVG.append("svg").attr("id", svgid + "innerSVG");

		// childSVG.attr("width", svgWidth).attr("height",svgHeight);

		var g = childSVG.append("g");
		g.attr("width", "100%").attr("height", "100%").selectAll("text").data(
				words).enter().append("text").style("font-size", function(d) {
			return d.size + "px";
		}).style("font-family", "Impact").style("fill", function(d, i) {
			return fill(d.color);
		}).style("cursor", function() {
			if (isDetailed == 1)
				return "pointer";
			else
				return "pointer";
		}).attr("text-anchor", "middle").attr("transform", function(d) {
			var xc = d.x;
			var yc = d.y;

			return "translate(" + [ xc, yc ] + ")rotate(" + d.rotate + ")";
		})
		// .attr("textLength","100")
		// .attr("lengthAdjust","spacingAndGlyphs")
		.text(function(d) {
			return d.text;
		});

		if (isDetailed == 1) {
			g.selectAll("text").on("click", function(d) {
//				ajaxGetSnippet("ajaxSnippetArea", "ajaxQueryDiv", d.text, oid);
				getSnippetAjax('snippets', d.text, oid, 0, rows4ngrams);
			});
		} else {
			g.selectAll("text").on("click", function(d) {
				// var mainqueryT = document.getElementById("mainqtextbox");
				// var curquery = mainqueryT.value;
				// var querywordArr = curquery.split(" ");
				// if( querywordArr.indexOf(d.label) == -1) {
				// querywordArr.push('"'+d.text+'"');
				// }
				// var newQuery = querywordArr.join("+");
				var newQuery = d.text;

				location.href = "collection1/s?q=" + newQuery + "&t=l";
			})

		}

		innerSVG = document.getElementById(svgid + "innerSVG");
		outterSVG = document.getElementById(svgid);

		svgWidth = innerSVG.getBBox().width * 1.1;
		svgHeight = innerSVG.getBBox().height * 1.1;

// force input
svgWidth = (!isDetailed && svgWidth == 0) ? 330 : svgWidth;
svgHeight = (!isDetailed && svgHeight == 0) ? 220 : svgHeight;

console.log(svgid+' svgWidth=' +svgWidth + ' width=' + innerSVG.getBBox().width);
console.log(svgid+' svgHeight=' +svgHeight + ' height=' + innerSVG.getBBox().height);

		// childSVG.attr("transform",
		// "translate("+(svgWidth/2)+","+(svgHeight/2)+")");
		childSVG.attr("viewBox", "0 0 " + svgWidth + " " + svgHeight);

		g.attr("transform", "translate(" + (svgWidth / 2) + "," + (svgHeight / 2) + ")");
	}

	// d3.layout.cloud().size([ 400, 230 ]).words(words.map(function(d) {
	d3.layout.cloud().size([ (isDetailed? 400:300), 200 ]).words(words.map(function(d) {
		// console.log(d.freq);
		return {
			text : d.word,
			// size : (d.freq * 126) / freq_max,
			size : d.freq,
			rank : d.rank,
			color : d.type,
		};
	})).randomize(false)
	// .rotate(function() { return ~~(Math.random() * 2) * 90; })
	//
	// .rotate(function() {
	// return 0;
	// })
	.rotate(function(d) {
		// return 0;
		if (d.rank <= horizontalTopN) {
			return 0;
		} else {
			// alert('d.rank: ' + d.rank + ' d.text: ' + d.text + '
			// horizontalTopN: ' + horizontalTopN);
			angle = ~~(d.size / 2) * 90;
			if (angle % 180 == 0)
				angle = 0;
			// return angle;
			return 0;
		}
	}).font("Impact").fontSize(function(d) {
		return 50 * (d.size - minSize) / gap + 20;
		// return d.size;
	}).on("end", draw).start();

}

// //////////////////////////////////////////////////////////////////
// ////////////////////// //////////////////////////
// ////////////////////// Node info //////////////////////////
// ////////////////////// //////////////////////////
// //////////////////////////////////////////////////////////////////

function decodeHTML(str) {
	str = str.replace(/(&gt;)/g, ">");
	str = str.replace(/(&lt;)/g, "<");

	return str;
}

function addrow(table, h, v) {

	var row = table.insertRow(table.rows.length);
	var cell = row.insertCell(row.cells.length);
	var cell2 = row.insertCell(row.cells.length);
	cell.align = "right";
	cell.width = "20%";
	cell.style.verticalAlign = "top";
	cell.innerHTML = "<b>"+h+"</b>";
	cell2.innerHTML = v;

}

function fillGeneInfo(table, geneinfo) {

	// addrow(table, "Name:", geneinfo["info"]["name"]);
	addrow(table, "Type:", "Gene");
	addrow(table, "Gene symbol:", geneinfo["info"]["symbol"]);
	addrow(table, "Synonyms:", geneinfo["info"]["synonyms"]);
	addrow(table, "Other names:", geneinfo["info"]["othernames"]);

	// addrow(table, "External links:",
	// decodeHTML(geneinfo["info"]["extLink"]));
	if(geneinfo["info"]["extLink"] != 'null'){
		addrow(table, "External links:", geneinfo["info"]["extLink"]);
	}
}

function fillDrugInfo(table, druginfo) {
	// addrow(table, "Name:", druginfo["info"]["name"]);
	addrow(table, "Type:", "Drug");
	addrow(table, "Brand name:", druginfo["info"]["brand"]);
	addrow(table, "Synonyms:", druginfo["info"]["synonyms"]);
	addrow(table, "Description:", druginfo["info"]["desc"]);
	// addrow(table, "References:", decodeHTML(druginfo["info"]["ref"]));
// adding toxicity and pharmacodynamics if they are not empty
	if(druginfo["info"]["tox"] !== undefined && druginfo["info"]["tox"] != "") {
		addrow(table, "Toxicity:", druginfo["info"]["tox"]);
	}

	if(druginfo["info"]["dynamics"] !== undefined && druginfo["info"]["dynamics"] != "") {
		addrow(table, "Pharmacodynamics:", druginfo["info"]["dynamics"]);
	}
	addrow(table, "References:", druginfo["info"]["ref"]);
}

function fillDiseaseInfo(table, diseaseinfo) {
	// addrow(table, "Name:", diseaseinfo["info"]["name"]);
	addrow(table, "Type:", "Disease");
	addrow(table, "Description:", diseaseinfo["info"]["desc"]);
}

function fillPathwayInfo(table, pathwayinfo) {
	// addrow(table, "Name:", pathwayinfo["info"]["name"]);
	addrow(table, "Type:", "Pathway");
	addrow(table, "External link:", pathwayinfo["info"]["link"]);
}

function fillNodata(table) {
	var row = table.insertRow(table.rows.length);
	var cell = row.insertCell();

	cell.innerHTML = "No information available.";
}

function fillObjInfo(cellid, nodeinfo) {
	var parentCell = document.getElementById(cellid);

	var table = document.createElement("table");

/*
	if (nodeinfo["type"] == "gene") {
		fillGeneInfo(table, nodeinfo);
	} else if (nodeinfo["type"] == "disease") {
		fillDiseaseInfo(table, nodeinfo);
	} else if (nodeinfo["type"] == "drug") {
		fillDrugInfo(table, nodeinfo);
	} else if (nodeinfo["type"] == "pathway") {
		fillPathwayInfo(table, nodeinfo);
	} else {
		fillNodata(table);
	}
*/
	//addrow(table, "Name", nodeinfo["Name"]);
	addrow(table, "Type", nodeinfo["Type"]);
	for(fieldName in nodeinfo) {
		if(fieldName == "Name" || fieldName == "External links" || fieldName == "Type") {
			continue;
		}
		if(typeof nodeinfo[fieldName] == typeof []) {
			addrow(table, fieldName, nodeinfo[fieldName].join("; "));
		} else {
			addrow(table, fieldName, nodeinfo[fieldName]);
		}
	}
	if(nodeinfo["External links"] != undefined) {
		if( typeof nodeinfo["External links"] === 'string' ) {
			nodeinfo["External links"] = [ nodeinfo["External links"] ];
		}
		var linkstr = nodeinfo["External links"].join("&nbsp;&nbsp;&nbsp;");
		//var linkstrExclEntityName = linkstr.replace(">"+ nodeinfo["Name"] +"<", "><");
		var linkstrExclEntityName = replaceAll(linkstr, ">"+ nodeinfo["Name"] +"<", "><")
//		addrow(table, "External links", nodeinfo["External links"]);
		addrow(table, "External links", linkstrExclEntityName);
	}

	parentCell.appendChild(table);
}

// //////////////////////////////////////////////////////////////////
// ////////////////////// //////////////////////////
// ////////////////////// Keywords //////////////////////////
// ////////////////////// //////////////////////////
// //////////////////////////////////////////////////////////////////

function keywordTable(divid, keywordData, topN) {
	var keywordsArr = keywordData["keywords"]
	var numOfKeywords = keywordsArr.length;

	var rankNum = (topN > numOfKeywords) ? numOfKeywords : topN;
	var parentDiv = document.getElementById(divid);

	var orderList = document.createElement("ul");

	var item;

	for (i = 0; i < rankNum; i++) {
		item = document.createElement("li");
		item.innerHTML = keywordsArr[i]["keyword"];
		orderList.appendChild(item);
	}

	parentDiv.appendChild(orderList);
}

// //////////////////////////////////////////////////////////////////
// ////////////////////// //////////////////////////
// ////////////////////// Bubble //////////////////////////
// ////////////////////// //////////////////////////
// //////////////////////////////////////////////////////////////////

function drawBubbleChart(svgName, sourceData, topn) {

	// Returns a flattened hierarchy containing all leaf nodes under the root.

	var diameter = 500, svgWidth = 400, svgHeight = 250, format = d3
			.format(",d"), color = d3.scale.category20c();

	var bubble = d3.layout.pack().sort(null).size([ svgWidth, svgHeight ])
			.padding(1.5);

	d3.select(svgName).select("svg").remove();

	var svg = d3.select(svgName).append("svg:svg").attr("width", svgWidth)
			.attr("height", svgHeight).attr("class", "bubble");

	var myMouseOver = function() {
		var node = d3.select(this);
		var circle = node.select("circle");
		circle.style("stroke", "#C55A11");
		circle.style("fill", "#FBE5D6");
		node.style("cursor", "pointer");
	}
	var myMouseOut = function() {
		var circle = d3.select(this).select("circle");
		circle.style("stroke", "#FBE5D6");
		circle.style("fill", "white");
	}
	var myOnclick = function() {
		var group = d3.select(this);
		var text = group.select("text");
		var querytext = text.selectAll("title");

		// alert('"'+querytext.text()+'"');
		// alert(sourceData["oid"]);

//		ajaxGetSnippet("ajaxSnippetArea", "ajaxQueryDiv", '"'+ querytext.text() + '"', sourceData["oid"]);
		getSnippetAjax('snippets', '"'+ querytext.text() + '"', sourceData["oid"], 0, rows4ngrams);
	}

	update(sourceData);

	function makeData(srcData) {

		var keywordList = srcData["keywords"];

		var resData = [];
		var minsize, maxsize;

		var bubbleCount = (topn < keywordList.length) ? topn
				: keywordList.length;

		maxsize = minsize = parseInt(keywordList[0]["count"]);
		for (i = 1; i < bubbleCount; i++) {
			if (parseInt(keywordList[i]["count"]) < minsize) {
				minsize = parseInt(keywordList[i]["count"]);
			} else if (parseInt(keywordList[i]["count"]) > maxsize) {
				maxsize = parseInt(keywordList[i]["count"]);
			}
		}

		if (minsize == maxsize)
			maxsize = minsize + 1;

		for (i = 0; i < bubbleCount; i++) {
			resData.push({
				textarr : keywordList[i]["keyword"].split(' '),
				value : (parseInt(keywordList[i]["count"]) - minsize)
						/ (maxsize - minsize) * 100 + 25
			});
		}

		return {
			children : resData
		};
	}

	function update(wordList) {
		var gnodes = svg.selectAll("g.node");

		var node = gnodes.data(
				bubble.nodes(makeData(wordList)).filter(function(d) {

					return !d.children;
				})).enter().append("svg:g").attr("class", "node").attr(
				"transform", function(d) {
					//console.log(d.x);
					return "translate(" + d.x + "," + d.y + ")";
				}).on("mouseover", myMouseOver).on("mouseout", myMouseOut).on(
				"click", myOnclick);

		node.append("svg:circle").attr("r", function(d) {
			return d.r;
		}).style("stroke", "#FBE5D6").style("stroke-width", 3).style("fill",
				function(d) {
					// return color(d.color);
					return "white";
				}).attr("title", function(d) {
			return d.textarr.join(" ");
		});

		node.append("svg:text").attr("dy", function(d) {
			if (d.textarr[3]) {
				return "-9px";
			} else if (d.textarr[2]) {
				return "-6px";
			} else {
				return "0px";
			}
		}).attr("fill", "black").style("text-anchor", "middle").text(
				function(d) {
					return d.textarr[0];
				}).style("font-size", function(d) {
			return Math.max(12, d.r / 3) + "px";
		}).style("font-weight", "bold").append("title").text(function(d) {
			return d.textarr.join(" ");
		});

		node.append("svg:text").attr("dy", function(d) {
			if (d.textarr[3]) {
				return "3px";
			} else if (d.textarr[2]) {
				return "6px";
			} else {
				return "12px";
			}
		}).attr("fill", "black").style("text-anchor", "middle").text(
				function(d) {
					return d.textarr[1];
				}).style("font-size", function(d) {
			return Math.max(12, d.r / 3) + "px";
		}).style("font-weight", "bold").append("title").text(function(d) {
			return d.textarr.join(" ");
		});

		node.append("svg:text").attr("dy", function(d) {
			if (d.textarr[3]) {
				return "15px";
			} else {
				return "18px";
			}
		}).attr("fill", "black").style("text-anchor", "middle").text(
				function(d) {
					if (d.textarr[2])
						return d.textarr[2];
					else
						return "";
				}).style("font-size", function(d) {
			return Math.max(12, d.r / 3) + "px";
		}).style("font-weight", "bold").append("title").text(function(d) {
			return d.textarr.join(" ");
		});

		node.append("svg:text").attr("dy", function(d) {
			return "26px";
		}).attr("fill", "black").style("text-anchor", "middle").text(
				function(d) {
					if (d.textarr[3])
						return d.textarr[3];
					else
						return "";
				}).style("font-size", function(d) {
			return Math.max(12, d.r / 3) + "px";
		}).style("font-weight", "bold").append("title").text(function(d) {
			return d.textarr.join(" ");
		});
	}
}

// //////////////////////////////////////////////////////////////////
// ////////////////////////////////////////////////
// //////////////////////Bar chart //////////////////////////
// ////////////////////////////////////////////////
// //////////////////////////////////////////////////////////////////
function drawBar(divid, sourceData) {

	var width = document.getElementById(divid).offsetWidth;
//	var height = document.getElementById(divid).offsetHeight;
	var height = 70;

	var totalSize = 0;
	var positions = [ 0 ];
	var curPortion = 0;

	var transformed = [];

	var etcList = [];

	var prePickedcolors = d3.scale.category20();
	var colors = {};
/*
	colors["Gene"] = prePickedcolors(0);

	colors["Drug"] = prePickedcolors(1);
	colors["Chemical Compound"] = prePickedcolors(2);
	colors["Target"] = prePickedcolors(3);
	colors["Disease"] = prePickedcolors(4);
	colors["Toxin"] = prePickedcolors(5);
	colors["Transcription Factor"] = prePickedcolors(6);

	colors["etc."] = prePickedcolors(7);

	colors["miRNA"] = prePickedcolors(8);
	colors["Pathway"] = prePickedcolors(9);

	colors["Mutation"] = prePickedcolors(10);
	colors["Cell Line"] = prePickedcolors(11);
*/
/*
	colors["Gene"] = "#1F77B4";

	colors["Drug"] = "#FF7F0E";
	colors["Chemical Compound"] = "#2CA02C";
	colors["Target"] = "#D62728";
	colors["Disease"] = "#9467BD";
	colors["Toxin"] = "#8C564B";
	colors["Transcription Factor"] = "#E377C2";

	colors["etc."] = "#7F7F7F";

	colors["miRNA"] = "#BCBD22";
	colors["Pathway"] = "#17BECF";

	colors["Mutation"] = "#393B79";
	colors["Cell Line"] = "#D100C3";
*/
/*
	colors["Gene"] = "#1ABC9C";

	colors["Drug"] = "#2ECC71";
	colors["Chemical Compound"] = "#3498D8";
	colors["Target"] = "#9B59B6";
	colors["Disease"] = "#34495E";
	colors["Toxin"] = "#F1C40F";
	colors["Transcription Factor"] = "#E67E22";

	colors["etc."] = "#95A5A6";

	colors["miRNA"] = "#418CA6";
	colors["Pathway"] = "#E74C3C";

	colors["Mutation"] = "#F39C12";
	colors["Cell Line"] = "#C0392B";
*/

	// colors["Gene"] = "#DB4251";
	colors["Gene/Protein"] = "#DB4251";

	colors["Drug"] = "#54B2B6";
	colors["Chemical Compound"] = "#8AAF6D";
	colors["Target"] = "#F5B939";
	colors["Disease"] = "#927F99";
	colors["Toxin"] = "#8DC63F";
	colors["Transcription Factor"] = "#96AFFF";

	colors["etc."] = "#1A97C0";

	colors["miRNA"] = "#E94465";
	colors["Pathway"] = "#F4DC39";

	colors["Mutation"] = "#259E78";
	colors["Cell Line"] = "#259E78";


	var dummy = d3.scale.category20b();
	var dummy2;
	for (i = 0; i < 20; i++) {
		dummy2 = dummy(i);
	}

//	colors["sel"] = dummy(4);
	colors["sel"] = "#777777";

	var divarea = d3.select("#" + divid);
	var svg = divarea.append("svg").attr("width",width+"px").attr("height",height+"px");

	var lines = [];
	var text1s = [];
	var text2s = [];
	var etctexts = [];
	var etcline;


	sourceData.sort(function(a, b) {
		return b.size - a.size;
	});

	for (i = 0; i < sourceData.length; i++) {
		totalSize += sourceData[i].size;
		positions.push(totalSize);
	}

	for (i = 0; i < sourceData.length; i++) {
		prevPortion = positions[i] / totalSize;
		curPortion = positions[i + 1] / totalSize;
		if (curPortion < 0.9) {
			transformed.push({
				"start" : positions[i] / totalSize * width,
				"end" : positions[i + 1] / totalSize * width,
				"name" : sourceData[i].name,
				"size" : sourceData[i].size,
				"url" : sourceData[i].url,
				"portion" : (sourceData[i].size / totalSize * 100).toFixed(1)
			});
		} else {
			transformed.push({
				"start" : positions[i] / totalSize * width,
				"end" : width,
				"name" : "etc.",
				"size" : totalSize - positions[i],
				"url" : "",
				"portion" : ((1 - prevPortion) * 100).toFixed(1)
			});

			for (j = i; j < sourceData.length; j++) {
				curPortion = positions[j + 1] / totalSize;
				etcList.push({
					"start" : positions[j] / totalSize * width,
					"end" : positions[j + 1] / totalSize * width,
					"name" : sourceData[j].name,
					"size" : sourceData[j].size,
					"url" : sourceData[j].url,
					"portion" : (sourceData[j].size / totalSize * 100)
							.toFixed(1)
				});
			}

			break;
		}

	}

	for (i = 0; i < transformed.length; i++) {
		var start = transformed[i].start;
		var end = transformed[i].end;

		var line = svg.append("svg:line").attr("x1", start).attr("x2", end)
				.attr("y1", 20).attr("y2", 20).style("stroke-width", 40).style(
						"stroke", colors[transformed[i].name]).attr("id", i).on("click", function() {
					if (transformed[this.id].url != "")
						location.href = (transformed[this.id].url);
				}).on("mouseover",
						function() {
						if(this.id != transformed.length-1)
							this.style.cursor = "pointer";
					mouseOverHandler(this.id, lines[this.id],text1s[this.id],text2s[this.id]);
				}).on("mouseout", function() {
			mouseOutHandler(this.id, lines[this.id],text1s[this.id],text2s[this.id]);
		});
		lines.push(line);
	}

	etcline = svg.append("svg:line").attr("x1", 0).attr("x2", end).attr("y1",
			55).attr("y2", 55).style("stroke-width", 30).style("stroke",
			colors["etc."])
			.on("mouseover", function() {
				var lid = transformed.length-1;
				mouseOverHandler(lid, lines[lid],text1s[lid],text2s[lid]);
	}).on("mouseout", function() {
		var lid = transformed.length-1;
		mouseOutHandler(lid, lines[lid],text1s[lid],text2s[lid]);
	})
	.attr("id", i).on("click", function() {
		if (transformed[this.id].url != "")
			location.href = (transformed[this.id].url);
	});

	for (i = 0; i < transformed.length; i++) {
		var start = transformed[i].start;
		var end = transformed[i].end;
		var name = transformed[i].name;
		var url = transformed[i].url;
		var portion = transformed[i].portion;
		var size = transformed[i].size;

		var text = svg.append("svg:text").style("text-anchor", "middle").attr(
				"dy", 15).attr("dx", (start + end) / 2).attr("font-size", 14)
				.attr("font-weight", "normal").style("fill", "white").text(name)
				.on("mouseover", function() {
					if(this.id != transformed.length-1)
					this.style.cursor = "pointer";
					mouseOverHandler(this.id, lines[this.id],text1s[this.id],text2s[this.id]);
				}).on("mouseout", function() {
					mouseOutHandler(this.id, lines[this.id],text1s[this.id],text2s[this.id]);
				}).attr("id", i).on("click", function() {
					if (transformed[this.id].url != "")
						location.href = (transformed[this.id].url);
				});

		var text2 = svg.append("svg:text").style("text-anchor", "middle").attr(
				"dy", 30).attr("dx", (start + end) / 2).attr("font-size", 11)
				.style("fill", "white")
				.text("(" + size + ", " + portion + "%)").on("mouseover",
						function() {
							if(this.id != transformed.length-1)
								this.style.cursor = "pointer";
					mouseOverHandler(this.id, lines[this.id],text1s[this.id],text2s[this.id]);
						}).on("mouseout", function() {
					mouseOutHandler(this.id, lines[this.id],text1s[this.id],text2s[this.id]);
				}).attr("id", i).on("click", function() {
					if (transformed[this.id].url != "")
						location.href = (transformed[this.id].url);
				});

		text1s.push(text);
		text2s.push(text2);
	}

	for (i = 0; i < etcList.length; i++) {

		var name = etcList[i].name;
		var portion = etcList[i].portion;
		var url = etcList[i].url;
		var size = etcList[i].size;

		var unit = width / etcList.length;
		var xpos = unit * (i + 1) - unit / 2;

		var text = svg.append("svg:text").style("text-anchor", "middle").attr(
				"dy", 60).attr("dx", xpos).style("fill", "white").text(
				name + " (" + size + ", " + portion + "%)").style("font-size",
				"12").on("mouseover", function() {
			this.style.cursor = "pointer";
			mouseOverHandler_etc(this.id, etctexts[this.id]);
		}).on("mouseout", function() {
			mouseOutHandler_etc(this.id, etctexts[this.id]);
		}).attr("id", i).on("click", function() {
			if (etcList[this.id].url != "")
				location.href = (etcList[this.id].url);
		});

		etctexts.push(text);
	}


	function mouseOverHandler(id, line, text1, text2) {
//		var line = lines[id];
//		var text1 = text1s[id];
//		var text2 = text2s[id];

		line.style("stroke", colors["sel"]);
		text1.style("font-weight", "bold");
		text2.style("font-weight", "bold");

		if (id == (lines.length - 1)) {
			etcline.style("stroke", colors["sel"]);
		}
	}


	function mouseOutHandler(id, line, text1, text2) {
//		var line = lines[id];
//		var text1 = text1s[id];
//		var text2 = text2s[id];

		var name = text1.text();

		line.style("stroke", colors[name]);
		text1.style("font-weight", "normal");
		text2.style("font-weight", "normal");

		if (id == (lines.length - 1)) {
			etcline.style("stroke", colors[name]);
		}
	}

	function mouseOverHandler_etc(id, etctext) {
//		var etctext = etctexts[id];
		var lid = transformed.length-1;
		mouseOverHandler(lid, lines[lid],text1s[lid],text2s[lid]);
		etctext.style("font-weight", "bold");
	}

	function mouseOutHandler_etc(id, etctext) {
		var etctext = etctexts[id];
		var lid = transformed.length-1;
		mouseOutHandler(lid, lines[lid],text1s[lid],text2s[lid]);
		etctext.style("font-weight", "normal");
	}

}

// //////////////////////////////////////////////////////////////////
// ////////////////////// //////////////////////////
// ////////////////////// Misc //////////////////////////
// ////////////////////// //////////////////////////
// //////////////////////////////////////////////////////////////////

function hitEnter(e, divid, querytextboxid, querytxt, oid) {
	if (e.keyCode == 13) {
		ajaxGetSnippet(divid, querytextboxid, querytxt, oid);
	} else {
		e.keyCode = 0;
		return;
	}
}

function shortenText(str, len) {
	var shortened = "";

	if (str.length > len) {
		shortened = str.substring(0, len);
		shortened += "...";
	} else {
		shortened = str;
	}

	return shortened;
}

function ajaxRequest() {
	if (window.XMLHttpRequest) // if Mozilla, Safari etc
	{
		return new XMLHttpRequest();
	} else if (window.ActiveXObject) { // Test for support for ActiveXObject in
		// IE (as XMLHttpRequest in IE7 is
		// broken)
		var activexmodes = [ "Msxml2.XMLHTTP", "Microsoft.XMLHTTP" ]; // activeX
		// versions
		// to
		// check
		// for
		// in IE

		for (var i = 0; i < activexmodes.length; i++) {
			try {
				return new ActiveXObject(activexmodes[i])
			} catch (e) {
				// suppress error
			}
		}
	} else
		return false;
}

var snippetStart = 0;
var numFound_total = 0;
var numPage = 0;
var rows = 200;
var monthNames = ["0", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

function ajaxGetSnippet(snippetdivid, querytextboxid, querytxt, oid) {
	var snippetArea = document.getElementById(snippetdivid);

	var queryTextBox = document.getElementById(querytextboxid);
	queryTextBox.value = querytxt;
	// var querytxt = queryTextBox.value;

	snippetStart = 0;

//	alert(navigator.userAgent);

	var mygetrequest = new ajaxRequest();

	mygetrequest.onreadystatechange = function() {
		if (mygetrequest.readyState == 4 && mygetrequest.status == 200) {

			snippetArea.innerHTML = "";

			var qtime = "";

			var xmldata = mygetrequest.responseXML;

			var lsts = xmldata.getElementsByTagName("lst");

			for(i=0;i< lsts.length;i++) {
				if(lsts[i].getAttribute("name") == 'responseHeader') {
					var headerVals = lsts[i].getElementsByTagName("int");
					if (headerVals.length > 0) {
						for (j = 0; j < headerVals.length; j++) {
							var arrname = headerVals[j].getAttribute("name");
							if (arrname == "QTime") {
								qtime = headerVals[j].firstChild.nodeValue;
								// alert(qtime);
								break;
							}
						}
					}
					break;
				}
			}

			var res = xmldata.documentElement.getElementsByTagName("result");
			var licnt = 0;

			if (res.length > 0) {
				var numFound = res[0].getAttribute("numFound");
				numFound_total = numFound;
				// snippetArea.innerHTML += "<span style='font-size: 13px'>Found
				// " + numFound + " mention(s)</span>";

				var div = Math.floor(numFound / 10);
				var rem = numFound % 10;
				if (rem > 0) {
					numPage = div + 1;
				} else {
					numPage = div;
				}
				snippetArea.innerHTML += "Page 1 (out of " + numPage
						+ " pages) [" + qtime + " ms]:";

				var ul = document.createElement("ul");
				var docs = res[0].getElementsByTagName("doc");
				var numFound = res[0].getAttribute("numFound");

				for (i = 0; i < docs.length; i++) {

					var pmid = "";
					var pmcid = "";
					var title = "";
					var journalTitle = "";
					var publishedMonth = "";
					var publishedYear = "";
					var contentStr = "";
					var content_highlightStr = "";

					for (j = 0; j < docs[i].childNodes.length; j++) {
						var node = docs[i].childNodes[j];

						// alert(pmid + "\n"+pmcid + "\n"+journalTitle +
						// "\n"+publishedYear + "\n"+contentStr);

						if (node.nodeName == "arr") {
							var arrname = node.getAttribute("name");
							if (arrname == "content" || arrname == "abstract") {
								contentStr += node.firstChild.firstChild.nodeValue;
							}
						} else if (node.nodeName == "str") {
							var stratt = node.getAttribute("name");

							if (stratt == "pid_pmc") {
								pmcid = node.firstChild.nodeValue;
							} else if (stratt == "pid_pmid") {
								pmid = node.firstChild.nodeValue;
							} else if (stratt == "title") {
								title = node.firstChild.nodeValue;
							} else if (stratt == "resourcename") {
								journalTitle = node.firstChild.nodeValue;
							} else if (stratt == "year") {
								publishedYear = node.firstChild.nodeValue;
							} else if (stratt == "text_highlight") {
								content_highlightStr = node.firstChild.nodeValue;
							}
						} else if (node.nodeName == "int") {
							var intatt = node.getAttribute("name");
							if (intatt == "month") {
								publishedMonth = monthNames[node.firstChild.nodeValue];
							}
						}
					}

					//
					// for(j = 0 ;j <arrs.length ; j++) {
					// if(arrs[j].getAttribute("name") == "content" ||
					// arrs[j].getAttribute("name") == "abstract") {
					// contentStr += arrs[j].firstChild.firstChild.nodeValue;
					//
					// }
					// }

//					if (contentStr.length > 600) {
//						contentStr = contentStr.substring(0, 600) + ' ...';
//					}
//
//					if (content_highlightStr.length > 600) {
//						content_highlightStr = content_highlightStr.substring(
//								0, 600)
//								+ ' ...';
//					}

					var li = document.createElement("li");
					// li.innerHTML = "<span style='font-size:
					// 15px;font-style:italic;color:brown;'>"+contentStr +
					// "</span><br/>" + journalTitle + ", " + publishedYear;
					li.innerHTML = "<span style='font-size: 15px;'>"
							+ content_highlightStr
							+ "</span><br/><span style='font-size: 14px;color:#6C3483;'>"
							+ title + "</span><br/><span style='font-size: 14px;font-style:italic;color:#5C4033;'>"
							+ journalTitle + "</span>, " + publishedMonth + " " + publishedYear + "&nbsp;";

					if (pmid != "") {
						var a_pmid = document.createElement("a");
						a_pmid.href = "https://www.ncbi.nlm.nih.gov/pubmed/"
								+ pmid;
						a_pmid.innerHTML = "[PubMed " + pmid + "]";
						a_pmid.target = "_blank";

						li.appendChild(a_pmid);
					}

					if (pmcid != "") {
						var a_pmcid = document.createElement("a");
						a_pmcid.href = "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC"
								+ pmcid + "/";
						a_pmcid.innerHTML = " [PMC " + pmcid + "]";
						a_pmcid.target = "_blank";

						li.appendChild(a_pmcid);
					}

					ul.appendChild(li);
					licnt++;
				}

				snippetArea.appendChild(ul);

				// draw new bubble
				var docs = mygetrequest.responseXML.documentElement.getElementsByTagName("doc");
				var ngramdata;
				for (var i = 0; i < docs.length; i++) {
					if (docs[i].getAttribute("name") == "NGrams") {
						ngramdata = docs[i].firstChild.firstChild.nodeValue;
					}
				}

				if (ngramdata) {
					drawBubbleChart("#bubblearea", eval('(' + ngramdata + ')'),
							10);
				}

				var querywordSpan = document.getElementById("snipetqueryword");
				querywordSpan.innerHTML = decodeURI(querytxt);
			} else {
				snippetArea.innerHTML = "No search results.";
			}

			if (licnt == 0) {
				snippetArea.innerHTML = "No search results.";
			}
		}
	}

	// mygetrequest.open("GET", "/v2/s?q=%2B%28abstract%3A%22" + querytxt +
	// "%22+content%3A%22" + querytxt + "%22%29+%2Boid%3A" + oid + "&t=m",
	// true);
//	 mygetrequest.open("GET", "/v2/s?q=%2Boid%3A" + oid +
//	 "+%2B%28abstract%3A%28" + querytxt
//	 + "%29+content%3A%28" + querytxt + "%29%29"
//	 + "&t=m&rows=" + rows + "&start=" + snippetStart, true);
//		mygetrequest.open("GET",
//				"./s?q=%2Boid%3A" + oid + "+%2Babstract%3A%28" + querytxt + "%29"
//						+ "&t=m&rows=" + rows + "&start=" + snippetStart, true);
//	mygetrequest.open("GET", "./s?q=abstract%3A%28" + encodeURI(querytxt) + "%29&oid=" + oid + "&t=m&rows=" + rows + "&start=" + snippetStart, true);
	mygetrequest.open("GET", "./s?q=abstract%3A%28" + encodeURI(querytxt) + "%29&fq=oid%3A" + oid + "&t=m&rows=" + rows + "&start=" + snippetStart, true);
	// mygetrequest.open("GET","/v2/select?q=oid:"+oid+"&t=m",true);
	mygetrequest.send(null);
}

function ajaxGetSnippetPaging(divid, querytextboxid, querytxt, oid, isNext) {
	if (isNext == true) {
		if (snippetStart == -1 || snippetStart + 10 > numFound_total) {
			return;
		}
		snippetStart += 10;
	} else {
		if (snippetStart == -1 || snippetStart == 0)
			return;
		snippetStart -= 10;
	}

	var snippetArea = document.getElementById(divid);
	var mygetrequest = new ajaxRequest();

	var queryTextBox = document.getElementById(querytextboxid);
	queryTextBox.value = querytxt;
	//
	// var querytxt = queryTextBox.value;

	mygetrequest.onreadystatechange = function() {
		if (mygetrequest.readyState == 4 && mygetrequest.status == 200) {

			snippetArea.innerHTML = "";

			var qtime = "";
			var xmldata = mygetrequest.responseXML;
			var lsts = xmldata.getElementsByTagName("lst");
			for(i=0;i< lsts.length;i++) {
				if(lsts[i].getAttribute("name") == 'responseHeader') {
					var headerVals = lsts[i].getElementsByTagName("int");
					if (headerVals.length > 0) {
						for (j = 0; j < headerVals.length; j++) {
							var arrname = headerVals[j].getAttribute("name");
							if (arrname == "QTime") {
								qtime = headerVals[j].firstChild.nodeValue;
								// alert(qtime);
								break;
							}
						}
					}
					break;
				}
			}

			var res = xmldata.documentElement.getElementsByTagName("result");
			var licnt = 0;

			if (res.length > 0) {
				// snippetArea.innerHTML += "<span style='font-size: 13px'>Found
				// " + numFound + " mention(s)</span><br />";

				snippetArea.innerHTML += "Page " + (snippetStart / 10 + 1)
						+ " (out of " + numPage + " pages) [" + qtime + " ms]:";

				var ul = document.createElement("ul");
				var docs = res[0].getElementsByTagName("doc");
				var numFound = res[0].getAttribute("numFound");
				var qtime = res[0].getAttribute("QTime");

				for (i = 0; i < docs.length; i++) {

					var pmid = "";
					var pmcid = "";
var title = "";
					var journalTitle = "";
					var publishedMonth = "";
					var publishedYear = "";
					var contentStr = "";
					var content_highlightStr = "";

					for (j = 0; j < docs[i].childNodes.length; j++) {
						var node = docs[i].childNodes[j];

						// alert(pmid + "\n"+pmcid + "\n"+journalTitle +
						// "\n"+publishedYear + "\n"+contentStr);

						if (node.nodeName == "arr") {
							var arrname = node.getAttribute("name");
							if (arrname == "content" || arrname == "abstract") {
								contentStr += node.firstChild.firstChild.nodeValue;
							}
						} else if (node.nodeName == "str") {
							var stratt = node.getAttribute("name");

							if (stratt == "pid_pmc") {
								pmcid = node.firstChild.nodeValue;
							} else if (stratt == "pid_pmid") {
								pmid = node.firstChild.nodeValue;
							} else if (stratt == "title") {
								title = node.firstChild.nodeValue;
							} else if (stratt == "resourcename") {
								journalTitle = node.firstChild.nodeValue;
							} else if (stratt == "year") {
								publishedYear = node.firstChild.nodeValue;
							} else if (stratt == "text_highlight") {
								content_highlightStr = node.firstChild.nodeValue;
							}
						}else if (node.nodeName == "int") {
							var intatt = node.getAttribute("name");
							if (intatt == "month") {
								publishedMonth = monthNames[node.firstChild.nodeValue];

							}
						}
					}

					//
					// for(j = 0 ;j <arrs.length ; j++) {
					// if(arrs[j].getAttribute("name") == "content" ||
					// arrs[j].getAttribute("name") == "abstract") {
					// contentStr += arrs[j].firstChild.firstChild.nodeValue;
					//
					// }
					// }

//					if (contentStr.length > 600) {
//						contentStr = contentStr.substring(0, 600) + ' ...';
//					}
//
//					if (content_highlightStr.length > 600) {
//						content_highlightStr = content_highlightStr.substring(
//								0, 600)
//								+ ' ...';
//					}

					var li = document.createElement("li");
					// li.innerHTML = "<span style='font-size:
					// 15px;font-style:italic;color:brown;'>"+contentStr +
					// "</span><br/>" + journalTitle + ", " + publishedYear;
					li.innerHTML = "<span style='font-size: 15px;'>"
							+ content_highlightStr
							+ "</span><br/><span style='font-size: 14px;color:#6C3483;'>"
							+ title + "</span><br/><span style='font-size: 14px;font-style:italic;color:#5C4033;'>"
							+ journalTitle + "</span>, " + publishedMonth + " " + publishedYear + "&nbsp;";

					if (pmid != "") {
						var a_pmid = document.createElement("a");
						a_pmid.href = "https://www.ncbi.nlm.nih.gov/pubmed/"
								+ pmid;
						a_pmid.innerHTML = "[PubMed " + pmid + "]";
						a_pmid.target = "_blank";

						li.appendChild(a_pmid);
					}

					if (pmcid != "") {
						var a_pmcid = document.createElement("a");
						a_pmcid.href = "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC"
								+ pmcid + "/";
						a_pmcid.innerHTML = " [PMC " + pmcid + "]";
						a_pmcid.target = "_blank";

						li.appendChild(a_pmcid);
					}

					ul.appendChild(li);
					licnt++;
				}

				snippetArea.appendChild(ul);

				var docs = mygetrequest.responseXML.documentElement
						.getElementsByTagName("doc");
				var ngramdata;
				for (var i = 0; i < docs.length; i++) {
					if (docs[i].getAttribute("name") == "NGrams") {
						ngramdata = docs[i].firstChild.firstChild.nodeValue;
					}
				}

				if (ngramdata)
					drawBubbleChart("#bubblearea", eval('(' + ngramdata + ')'),
							10);

				var querywordSpan = document.getElementById("snipetqueryword");
				querywordSpan.innerHTML = decodeURI(querytxt);
			} else {
				snippetArea.innerHTML = "No search results.";
			}

			if (licnt == 0) {
				snippetArea.innerHTML = "No search results.";
			}

		}

	}

	// mygetrequest.open("GET", "/v2/s?q=%2B%28abstract%3A%22" + querytxt +
	// "%22+content%3A%22" + querytxt + "%22%29+%2Boid%3A" + oid + "&t=m",
	// true);
	// mygetrequest.open("GET", "/v2/s?q=%2Boid%3A" + oid +
	// "+%2B%28abstract%3A%28" + querytxt
	// + "%29+content%3A%28" + querytxt + "%29%29"
	// + "&t=m&rows=" + rows + "&start=" + snippetStart, true);
//	mygetrequest.open("GET",
//			"./s?q=%2Boid%3A" + oid + "+%2Babstract%3A%28" + querytxt + "%29"
//					+ "&t=m&rows=" + rows + "&start=" + snippetStart, true);
//	mygetrequest.open("GET", "./s?q=abstract%3A%28" + encodeURI(querytxt) + "%29&oid=" + oid + "&t=m&rows=" + rows + "&start=" + snippetStart, true);
	mygetrequest.open("GET", "./s?q=abstract%3A%28" + encodeURI(querytxt) + "%29&fq=oid%3A" + oid + "&t=m&rows=" + rows + "&start=" + snippetStart, true);
	// mygetrequest.open("GET","/v2/select?q=oid:"+oid+"&t=m",true);
	mygetrequest.send(null);
}

function writeLink2TopObject() {
	var linkDiv = document.getElementById("links");

	var oNames = $("a.rankNobjectName");
	if(oNames != null && oNames.length > 0){
		var links = "Move to ";
		for (var i = 0; i < oNames.length; i++) {
			//console.log(oNames[i].id);
			if(i>0){
				links += "&nbsp;&nbsp;";
			}
			links += oNames[i].id.replace("obj", "") + ". " + "<a href='#" + oNames[i].id + "'>" +  oNames[i].text + "</a>";
		}
		linkDiv.innerHTML = links;
	}
}

function writeLinkNavButtons(start, rows, numFound, params) {
	getPagination("links3", start, rows, numFound, params);
}

function getPagination(paginationDivID, start, rows, numFound, params) {
	var paginationDiv = document.getElementById(paginationDivID);
	if(typeof paginationDiv != 'undefined'){
		var pages = "";

		var params_prefix;
		var params_suffix;

		var start_stt = params.indexOf("&start=");
		var start_end = -1;
		if(start_stt > -1){
			params_prefix = params.substring(0, start_stt);
			start_end = params.indexOf("&", start_stt+1);
			if(start_end > -1){
				params_suffix = params.substring(start_end+1);
			}else{
				params_suffix = "";
			}
		}else{
			params_prefix = params;
			params_suffix = "";
		}

//		console.log("params_prefix: " + params_prefix);
//		console.log("params_suffix: " + params_suffix);

		if(start > 0){
			var img_button_prev = document.createElement("img");
			img_button_prev.src = "/img/arrow_left.jpg";
			img_button_prev.onclick = function() {goBack()};
			img_button_prev.style.height = "25px";
			img_button_prev.style.width = "25px";
			img_button_prev.style.verticalAlign = "middle";
			img_button_prev.style.cursor = "pointer";
//			paginationDiv.appendChild(img_button_prev);
//			paginationDiv.innerHTML += "&nbsp;&nbsp;";

			pages += "<a href='./s?" +  params_prefix + "&start=" + (start-rows)  + params_suffix + "' style='font-weight: normal; vertical-align: middle;'><img src='/img/arrow_left.jpg' style='height: 25px; width: 25px; vertical-align: middle; cursor: pointer;'/>&nbsp;Prev.</a>&nbsp;&nbsp;";
		}

		var p_current = start / rows + 1;
		var p_total = Math.ceil(numFound / rows);

//		console.log(p_current);
//		console.log(p_total);

		var p_start = 1;
		var p_end = 10;

		if(p_total >= 10){
			if(p_current <= 6){	// the first page list
				p_start = 1;
				p_end = 10;
			}else if (p_current + 5 > p_total){ // the last page list
				p_start =  p_total - 9;
				p_end =  p_total;
			}else{
				p_start =  p_current - 5;
				p_end =  p_current + 4;
			}
		}else{
			p_start =  1;
			p_end =  p_total;
		}

//		console.log(p_start);
//		console.log(p_end);

		for (var i = p_start; i <= p_end; i++) {
			if(i>p_start){
				pages += "&nbsp;&nbsp;";
			}

			if(p_current == i){
				pages += "<b style='vertical-align: middle;'>" + i + "</b>";
			}else{
				pages += "<a href='./s?" + params_prefix + "&start=" + (i-1)*rows  + params_suffix + "' style='font-weight: normal; vertical-align: middle;'>" + i + "</a>";
			}
		}

		if(start+rows <= numFound){
			pages += "&nbsp;&nbsp;<a href='./s?" + params_prefix + "&start=" + (p_current*rows)  + params_suffix + "' style='font-weight: normal; vertical-align: middle;'>Next&nbsp;<img src='/img/arrow_right.jpg' style='height: 25px; width: 25px; vertical-align: middle;'/></a>";
		}

		paginationDiv.innerHTML += pages;
	}
}

var start_current = 0;
var rows4ngrams = 200;

function getSnippetAjax(snippetDivID, queryText, oid, start) {
	start_current = start;

	queryText = queryText.replace(/%22/g, "\"");

	document.getElementById('ajaxQueryDiv').value = queryText;

	var mygetrequest = new ajaxRequest();
	mygetrequest.onreadystatechange = function() {
		if (mygetrequest.readyState == 4 && mygetrequest.status == 200) {
			var xmldata = mygetrequest.responseXML;
			var lsts = xmldata.getElementsByTagName("lst");

			var qtime = "";
			for(i=0;i< lsts.length;i++) {
				if(lsts[i].getAttribute("name") == 'responseHeader') {
					var headerVals = lsts[i].getElementsByTagName("int");
					if (headerVals.length > 0) {
						for (j = 0; j < headerVals.length; j++) {
							var arrname = headerVals[j].getAttribute("name");
							if (arrname == "QTime") {
								qtime = headerVals[j].firstChild.nodeValue;
								// alert(qtime);
								break;
							}
						}
					}
					break;
				}
			}

			var res = xmldata.documentElement.getElementsByTagName("result");
			var licnt = 0;

			if (res.length > 0) {
				var snippetArea = document.getElementById(snippetDivID);
				snippetArea.innerHTML = "";

				var numFound = res[0].getAttribute("numFound");

				var start_show = start+1;
				if(numFound == 0){
					start_show = 0;
				}

				var end_show = start + 10;
				if(end_show > numFound){
					end_show = numFound;
				}

				var snippetArea = document.getElementById(snippetDivID);
				snippetArea.innerHTML = "";
				snippetArea.innerHTML += "Hits " + start_show + " - " + end_show + " [out of " + numFound + " matching mentions] (" + qtime + " ms):";
				snippetArea.innerHTML += "<br/><br/><h3>Abstracts</h3>"

				var ul = document.createElement("ul");
				var docs = res[0].getElementsByTagName("doc");
				var numFound = res[0].getAttribute("numFound");

				for (i = 0; i < docs.length; i++) {
					var pmid = "";
					var pmcid = "";
					var title = "";
					var journalTitle = "";
					var publishedMonth = "";
					var publishedYear = "";
					var contentStr = "";
					var content_highlightStr = "";

					for (j = 0; j < docs[i].childNodes.length; j++) {
						var node = docs[i].childNodes[j];

						// console.log(pmid + "\n"+pmcid + "\n"+journalTitle + "\n" + publishedYear + "\n"+contentStr);

						if (node.nodeName == "arr") {
							var arrname = node.getAttribute("name");
							if (arrname == "content" || arrname == "abstract") {
								contentStr += node.firstChild.firstChild.nodeValue;
							}
						} else if (node.nodeName == "str") {
							var stratt = node.getAttribute("name");

							if (stratt == "pid_pmc") {
								pmcid = node.firstChild.nodeValue;
							} else if (stratt == "pid_pmid") {
								pmid = node.firstChild.nodeValue;
							} else if (stratt == "title") {
								title = node.firstChild.nodeValue;
							} else if (stratt == "resourcename") {
								journalTitle = node.firstChild.nodeValue;
							} else if (stratt == "year") {
								publishedYear = node.firstChild.nodeValue;
							} else if (stratt == "text_highlight") {
								content_highlightStr = node.firstChild.nodeValue;
							} else if (stratt == "abstract_highlight") {
								content_highlightStr = node.firstChild.nodeValue;
							}
						} else if (node.nodeName == "int") {
							var intatt = node.getAttribute("name");
							if (intatt == "month") {
								publishedMonth = monthNames[node.firstChild.nodeValue];
							}
						}
					}

					var li = document.createElement("li");
					li.innerHTML = "<span style='font-size: 15px;'>"
							+ content_highlightStr
							+ "</span><br/><span style='font-size: 14px;color:#6C3483;'>"
							+ title + "</span><br/><span style='font-size: 14px;font-style:italic;color:#5C4033;'>"
							+ journalTitle + "</span>, " + publishedMonth + " " + publishedYear;

					if (pmid != "") {
						li.innerHTML += "&nbsp;";
						var a_pmid = document.createElement("a");
						a_pmid.href = "https://www.ncbi.nlm.nih.gov/pubmed/" + pmid;
						a_pmid.innerHTML = "[PubMed " + pmid + "]";
						a_pmid.target = "_blank";
						li.appendChild(a_pmid);
					}

					if (pmcid != "") {
						li.innerHTML += "&nbsp;";
						var a_pmcid = document.createElement("a");
						a_pmcid.href = "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC" + pmcid + "/";
						a_pmcid.innerHTML = "[PMC " + pmcid + "]";
						a_pmcid.target = "_blank";
						li.appendChild(a_pmcid);
					}

					ul.appendChild(li);
					licnt++;
				}

				snippetArea.appendChild(ul);

				// pagination
				getPaginationOfAjax(snippetDivID, encodeURI(queryText), 'pagination', oid, start, 10, numFound);

				// draw new bubble
				var docs = xmldata.documentElement.getElementsByTagName("doc");
				var ngramdata;
				for (var i = 0; i < docs.length; i++) {
					if (docs[i].getAttribute("name") == "NGrams") {
						ngramdata = docs[i].firstChild.firstChild.nodeValue;
					}
				}

				if (ngramdata) {
					drawBubbleChart("#bubblearea", eval('(' + ngramdata + ')'), 10);
				}

				var querywordSpan = document.getElementById("snipetqueryword");
				querywordSpan.innerHTML = decodeURI(queryText);
			} else {
				snippetArea.innerHTML = "No search results.";
			}

			if (licnt == 0) {
				snippetArea.innerHTML = "No search results.";
			}
		}
	}

//	console.log("getSnippetAjax: q=" + encodeURI(queryText) + " start=" + start);
	mygetrequest.open("GET", "./s?q=abstract%3A%28" + encodeURI(queryText) + "%29&fq=oid%3A" + oid + "&t=m&rows=" + rows4ngrams + "&start=" + start, true);
	mygetrequest.send(null);
}

function getPaginationOfAjax(snippetDivID, queryText, paginationDivID, oid, start, rows, numFound) {
	var paginationDiv = document.getElementById(paginationDivID);
	if(typeof paginationDiv != 'undefined'){
//		paginationDiv.innerHTML = "";
//		paginationDiv.reset();
//		while (paginationDiv.firstChild) {
//			paginationDiv.removeChild(paginationDiv.firstChild);
//		}

		var decodedQ = decodeURI(queryText);
//		console.log(decodedQ);
		decodedQ = decodedQ.replace(/\"/g, "%22");
//		console.log(decodedQ);

		var pages = "";

		var jsPrefix = "getSnippetAjax('" + snippetDivID + "', '" + decodedQ + "', " + oid + ", ";
//		console.log(jsPrefix);

		if(start > 0){
			var img_button_prev = document.createElement("img");
			img_button_prev.src = "/img/arrow_left.jpg";
			img_button_prev.onclick = function() {goBack()};
			img_button_prev.style.height = "25px";
			img_button_prev.style.width = "25px";
			img_button_prev.style.verticalAlign = "middle";
			img_button_prev.style.cursor = "pointer";
//			paginationDiv.appendChild(img_button_prev);
//			paginationDiv.innerHTML += "&nbsp;&nbsp;";

			var snippetJS = jsPrefix + (start-10) + "); document.getElementById('ajaxQueryDiv').focus(); return false;";
//			console.log(snippetJS);
			pages += "<a href=\"#\" onclick=\"" +snippetJS +  "\" style=\"font-weight: normal; vertical-align: middle;\"><img src=\"/img/arrow_left.jpg\" style=\"height: 25px; width: 25px; vertical-align: middle; cursor: pointer;\"/>&nbsp;Prev.</a>&nbsp;&nbsp;";
		}

		var p_current = start / rows + 1;
		var p_total = Math.ceil(numFound / rows);

//		console.log(p_current);
//		console.log(p_total);

		var p_start = 1;
		var p_end = 10;

		if(p_total >= 10){
			if(p_current <= 6){	// the first page list
				p_start = 1;
				p_end = 10;
			}else if (p_current + 5 > p_total){ // the last page list
				p_start =  p_total - 9;
				p_end =  p_total;
			}else{
				p_start =  p_current - 5;
				p_end =  p_current + 4;
			}
		}else{
			p_start =  1;
			p_end =  p_total;
		}

//		console.log(p_start);
//		console.log(p_end);


		for (var i = p_start; i <= p_end; i++) {
			if(i>p_start){
				pages += "&nbsp;&nbsp;";
				paginationDiv.innerHTML += "&nbsp;&nbsp;";
			}

			if(p_current == i){
				pages += "<b style='vertical-align: middle;'>" + i + "</b>";

//				var a_page_current = document.createElement("b");
//				a_page_current.style.verticalAlign = "middle";
//				a_page_current.innerHTML = i;
//				paginationDiv.appendChild(a_page_current);
			}else{
				var snippetJS = jsPrefix + (i-1)*rows + "); document.getElementById('ajaxQueryDiv').focus(); return false;";
//				console.log(snippetJS);
				pages += "<a href=\"#\" onclick=\"" + snippetJS + "\" style=\"font-weight: normal; vertical-align: middle;\">" + i + "</a>";

//				console.log(snippetDivID + " " + queryText + " " + oid+ " " + rows+ " " + i);


//				var pageStart = (i-1)*rows;


//				console.log("["+i+"] "+snippetDivID + " " + decodedQ + " " + oid+ " " + rows+ " " + pageStart);

//				var a_pageLink = document.createElement("a");
//				a_pageLink.href = "";
//				a_pageLink.onclick = function() {
////					getSnippetAjax(snippetDivID, decodedQ, oid, pageStart);
////					document.getElementById('ajaxQueryDiv').focus();
//				};
//				a_pageLink.style.fontWeight = "normal";
//				a_pageLink.style.verticalAlign = "middle";
//				a_pageLink.innerHTML = i;
//				console.log(a_pageLink);
//				console.log(a_pageLink.onclick);
//				paginationDiv.appendChild(a_pageLink);
			}
		}

//		var paginationTestDiv = document.getElementById('pagination_test');
//		for ( var d = p_start; d <= p_end; d++ ) (function(d){
//			if(i>p_start){
//				paginationTestDiv.innerHTML += "&nbsp;&nbsp;";
//			}
//
//			var pageStart = (d-1)*rows;
//
//			console.log("["+d+"] "+snippetDivID + " " + decodedQ + " " + oid+ " " + rows+ " " + pageStart);
//
//			var a_pageLink = document.createElement("a");
//			a_pageLink.href = "";
//			a_pageLink.onclick = function() {
//				alert("["+d+"] "+snippetDivID + " " + decodedQ + " " + oid+ " " + rows+ " " + pageStart);
//				getSnippetAjax(snippetDivID, decodedQ, oid, pageStart);
////				document.getElementById('ajaxQueryDiv').focus();
//			};
//			a_pageLink.style.fontWeight = "normal";
//			a_pageLink.style.verticalAlign = "middle";
//			a_pageLink.innerHTML = d;
//
////			console.log(a_pageLink);
//			paginationTestDiv.appendChild(a_pageLink);
//		})(d);

//		console.log(paginationDiv);

		if(start+rows <= numFound){
			var snippetJS = jsPrefix + (p_current*rows) + "); document.getElementById('ajaxQueryDiv').focus(); return false;";
//			console.log(snippetJS);
			pages += "&nbsp;&nbsp;<a href=\"#\" onclick=\"" + snippetJS + "\" style=\"font-weight: normal; vertical-align: middle;\">Next&nbsp;<img src=\"/img/arrow_right.jpg\" style=\"height: 25px; width: 25px; vertical-align: middle;\"/></a>";
		}

		paginationDiv.innerHTML = pages;
	}
}


function copyPagination(fromDivID, toDivID) {
	var paginationDiv = document.getElementById(toDivID);
	if(typeof paginationDiv != 'undefined'){
		paginationDiv.innerHTML = document.getElementById(fromDivID).innerHTML;
	}
}

function goBack() {
    javascript:history.back();
}

function replaceAll(str, searchStr, replaceStr) {
    return str.split(searchStr).join(replaceStr);
}

function resetRanges(solr, entitynum, quality, recency) {
    document.getElementById(solr).value = 1;
    document.getElementById(entitynum).value = 1;
    document.getElementById(quality).value = 1;
    document.getElementById(recency).value = 1;
}

function resetRanges2(quality, recency, qualityVal, recencyVal) {
    document.getElementById(quality).value = 1;
    document.getElementById(recency).value = 1;
    document.getElementById(qualityVal).value = 1;
    document.getElementById(recencyVal).value = 1;
}

function toggle_visibility(id) {
       var e = document.getElementById(id);
       if(e.style.display == 'block')
          e.style.display = 'none';
       else
          e.style.display = 'block';
}

function sosoojum(silsoo, jaree) {
	return silsoo.toFixed(jaree);
}

function toggle2(id, btn) {
	var obj = document.getElementById(id);
	var button = document.getElementById(btn);

	if(obj.style.display == 'block'){
		obj.style.display = 'none';
		button.innerHTML = 'Show info';
	} else {
		obj.style.display = 'block';
		button.innerHTML = 'Hide info';
	}

}
