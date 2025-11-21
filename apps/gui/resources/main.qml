import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Dialogs

ApplicationWindow {
    visible: true
    width: 1200
    height: 900
    title: "T2D-VAT Suite"

    property string previewSource: ""

    Popup {
        id: imagePopup
        anchors.centerIn: parent
        width: parent.width * 0.9
        height: parent.height * 0.9
        modal: true
        focus: true
        closePolicy: Popup.CloseOnEscape | Popup.CloseOnPressOutside

        background: Rectangle {
            color: "white"
            border.color: "#ccc"
            radius: 5
        }

        ColumnLayout {
            anchors.fill: parent
            anchors.margins: 10
            
            Image {
                Layout.fillWidth: true
                Layout.fillHeight: true
                fillMode: Image.PreserveAspectFit
                source: previewSource
                smooth: true
            }
            
            Button {
                text: "Close"
                Layout.alignment: Qt.AlignHCenter
                onClicked: imagePopup.close()
            }
        }
    }

    FileDialog {
        id: fileDialog
        title: "Select Input Excel File"
        nameFilters: ["Excel files (*.xlsx *.xls)", "All files (*)"]
        onAccepted: {
            viewModel.inputPath = selectedFile
        }
    }

    FolderDialog {
        id: folderDialog
        title: "Select Output Directory"
        onAccepted: {
            viewModel.outputDir = selectedFolder
        }
    }

    ColumnLayout {
        anchors.fill: parent
        spacing: 10

                // Header Section
                GroupBox {
                    title: "Configuration"
                    Layout.fillWidth: true
                    Layout.margins: 10

                    GridLayout {
                        columns: 3
                        rowSpacing: 10
                        columnSpacing: 10
                        width: parent.width

                        Label { text: "Input File:" }
                        TextField {
                            text: viewModel.inputPath
                            Layout.fillWidth: true
                            readOnly: true
                            placeholderText: "Select input Excel file..."
                        }
                        Button {
                            text: "Browse..."
                            onClicked: fileDialog.open()
                            enabled: !viewModel.isBusy
                        }

                Label { text: "Output Dir:" }
                TextField {
                    text: viewModel.outputDir
                    Layout.fillWidth: true
                    readOnly: true
                    placeholderText: "Select output directory..."
                }
                RowLayout {
                    spacing: 5
                    Button {
                        text: "Browse..."
                        onClicked: folderDialog.open()
                        enabled: !viewModel.isBusy
                    }
                    Button {
                        text: "Open Folder"
                        enabled: viewModel.outputDir !== ""
                        onClicked: viewModel.openOutputFolder()
                    }
                }

                Label { text: "Status:" }
                Label { 
                    text: viewModel.statusMessage 
                    Layout.fillWidth: true 
                    elide: Text.ElideRight
                    color: "#666"
                    font.italic: true
                }
                Item {
                    Layout.preferredWidth: 24
                    Layout.preferredHeight: 24
                    BusyIndicator {
                        anchors.centerIn: parent
                        width: 24
                        height: 24
                        running: viewModel.isBusy
                        visible: viewModel.isBusy
                    }
                }
            }
        }                TabBar {
                    id: bar
                    width: parent.width
                    
                    TabButton {
                        text: "Reproducer"
                        
                        contentItem: Text {
                            text: parent.text
                            font: parent.font
                            color: parent.checked ? "#2196F3" : "black"
                            horizontalAlignment: Text.AlignHCenter
                            verticalAlignment: Text.AlignVCenter
                            elide: Text.ElideRight
                        }

                        background: Rectangle {
                            color: parent.checked ? "white" : "#eee"
                            border.color: "#ccc"
                            Rectangle {
                                width: parent.width
                                height: 2
                                anchors.top: parent.top
                                color: parent.checked ? "#2196F3" : "transparent"
                            }
                        }
                    }
                    TabButton {
                        text: "Injector (ML)"
                        
                        contentItem: Text {
                            text: parent.text
                            font: parent.font
                            color: parent.checked ? "#2196F3" : "black"
                            horizontalAlignment: Text.AlignHCenter
                            verticalAlignment: Text.AlignVCenter
                            elide: Text.ElideRight
                        }

                        background: Rectangle {
                            color: parent.checked ? "white" : "#eee"
                            border.color: "#ccc"
                            Rectangle {
                                width: parent.width
                                height: 2
                                anchors.top: parent.top
                                color: parent.checked ? "#2196F3" : "transparent"
                            }
                        }
                    }
                }

                StackLayout {
                    width: parent.width
                    currentIndex: bar.currentIndex
                    Layout.fillHeight: true
                    Layout.margins: 10

                    // Reproducer Tab
                    Item {
                        ColumnLayout {
                            anchors.fill: parent
                            
                            RowLayout {
                                Label { text: "Top N Proteins:" }
                                SpinBox {
                                    from: 1
                                    to: 100
                                    value: viewModel.reproducerTopN
                                    onValueModified: viewModel.reproducerTopN = value
                                    editable: true
                                    enabled: !viewModel.isBusy
                                }
                                Button {
                                    text: "Run Reproducer"
                                    highlighted: true
                                    enabled: !viewModel.isBusy && viewModel.inputPath !== "" && viewModel.outputDir !== ""
                                    onClicked: viewModel.runReproducer()
                                }
                            }

                            ScrollView {
                                Layout.fillWidth: true
                                Layout.fillHeight: true
                                clip: true

                                Flow {
                                    width: parent.width
                                    spacing: 10
                                    leftPadding: Math.max(0, (width - (Math.floor((width + 10) / (550 + 10)) * (550 + 10) - 10)) / 2)
                                    
                                    Repeater {
                                        model: viewModel.reproducerImages
                                        delegate: Column {
                                            width: 550
                                            height: 450
                                            spacing: 5
                                            
                                            Image {
                                                source: modelData
                                                width: 550
                                                height: 400
                                                fillMode: Image.PreserveAspectFit
                                                cache: false 
                                                sourceSize.width: 550
                                                sourceSize.height: 400
                                                
                                                Rectangle {
                                                    anchors.fill: parent
                                                    color: "transparent"
                                                    border.color: "#ccc"
                                                }

                                                MouseArea {
                                                    anchors.fill: parent
                                                    cursorShape: Qt.PointingHandCursor
                                                    onClicked: {
                                                        previewSource = modelData
                                                        imagePopup.open()
                                                    }
                                                }
                                            }
                                            Text {
                                                text: modelData.split("/").pop()
                                                width: parent.width
                                                horizontalAlignment: Text.AlignHCenter
                                                elide: Text.ElideMiddle
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }

                    // Injector Tab
                    Item {
                        ColumnLayout {
                            anchors.fill: parent
                            
                            RowLayout {
                                Label { text: "Top Features:" }
                                SpinBox {
                                    from: 1
                                    to: 100
                                    value: viewModel.injectorTopN
                                    onValueModified: viewModel.injectorTopN = value
                                    editable: true
                                    enabled: !viewModel.isBusy
                                }
                                Button {
                                    text: "Run Injector"
                                    highlighted: true
                                    enabled: !viewModel.isBusy && viewModel.inputPath !== "" && viewModel.outputDir !== ""
                                    onClicked: viewModel.runInjector()
                                }
                            }

                            SplitView {
                                Layout.fillWidth: true
                                Layout.fillHeight: true
                                orientation: Qt.Horizontal

                                // Metrics Area
                                GroupBox {
                                    title: "Metrics"
                                    SplitView.preferredWidth: 300
                                    SplitView.minimumWidth: 200
                                    
                                    ScrollView {
                                        anchors.fill: parent
                                        TextArea {
                                    text: viewModel.injectorMetrics
                                    readOnly: true
                                    font.family: "Courier"
                                }
                            }
                        }                                // Plots Area
                                ScrollView {
                                    SplitView.fillWidth: true
                                    clip: true

                                    Flow {
                                        width: parent.width
                                        spacing: 10
                                        leftPadding: Math.max(0, (width - (Math.floor((width + 10) / (400 + 10)) * (400 + 10) - 10)) / 2)
                                        
                                        Repeater {
                                            model: viewModel.injectorImages
                                            delegate: Column {
                                                width: 400
                                                height: 350
                                                spacing: 5
                                                
                                                Image {
                                            source: modelData
                                            width: 400
                                            height: 300
                                            fillMode: Image.PreserveAspectFit
                                            cache: false
                                            sourceSize.width: 400
                                            sourceSize.height: 300

                                            Rectangle {
                                                anchors.fill: parent
                                                color: "transparent"
                                                border.color: "#ccc"
                                            }

                                            MouseArea {
                                                anchors.fill: parent
                                                cursorShape: Qt.PointingHandCursor
                                                onClicked: {
                                                    previewSource = modelData
                                                    imagePopup.open()
                                                }
                                            }
                                        }
                                                Text {
                                                    text: modelData.split("/").pop()
                                                    width: parent.width
                                                    horizontalAlignment: Text.AlignHCenter
                                                    elide: Text.ElideMiddle
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
    }
}
